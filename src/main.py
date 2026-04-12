"""
Image Watermarking CLI
Interactive command-line interface for watermarking, attacks, and evaluation
"""

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from pathlib import Path
import cv2
import os
import questionary
from questionary import Style

from attacks import (
    jpeg_compression,
    gaussian_noise,
    blurring,
    cropping,
    scaling
)
from evaluation import (
    calculate_psnr,
    calculate_ssim,
    calculate_ber,
    evaluate_watermark_robustness
)
from watermarking import (
    run_embedding_pipeline,
    extract_watermark,
    ALPHA,
)
from watermarking.extraction import (
    run_extraction_pipeline,
    extract_watermark_robust,
    extract_watermark_batch,
)
from watermarking.arnold import (
    arnold_transform,
    inverse_arnold_transform,
    arnold_period,
)

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_IMAGES_DIR = PROJECT_ROOT / "assets" / "input_images"
WATERMARKED_IMAGES_DIR = PROJECT_ROOT / "assets" / "watermarked_images"
WATERMARKS_DIR = PROJECT_ROOT / "assets" / "watermarks"
RESULTS_DIR = PROJECT_ROOT / "results"
ATTACK_RESULTS_DIR = RESULTS_DIR / "attack_results"
EXTRACTED_ATTACKED_WATERMARKS_DIR = RESULTS_DIR / "extracted_attacked_watermarks"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

SELECT_STYLE = Style(
    [
        ("question", "bold #8ecae6"),
        ("answer", "bold #90be6d"),
        ("pointer", "bold #ffb703"),
        ("highlighted", "bold #fb8500"),
        ("selected", "#90be6d"),
        ("instruction", "#6c757d"),
    ]
)

ATTACK_METHODS = {
    "1": {"name": "JPEG Compression", "func": jpeg_compression},
    "2": {"name": "Gaussian Noise", "func": gaussian_noise},
    "3": {"name": "Blurring", "func": blurring},
    "4": {"name": "Cropping", "func": cropping},
    "5": {"name": "Scaling", "func": scaling},
}

EVALUATION_METHODS = {
    "1": "PSNR (Peak Signal-to-Noise Ratio)",
    "2": "SSIM (Structural Similarity Index)",
    "3": "BER (Bit Error Rate)",
}


def clear_screen():
    """Clear terminal screen between menus on Windows."""
    os.system("cls")


def render_shell(title, subtitle=None):
    """Render a modern header panel for each screen."""
    header_text = Text(title, style="bold #8ecae6")
    if subtitle:
        header_text.append("\n")
        header_text.append(subtitle, style="#adb5bd")
    console.print(Panel(header_text, border_style="#219ebc", padding=(1, 2)))


def select_from_menu(title, options):
    """Arrow-key menu selection using up/down and Enter."""
    clear_screen()
    render_shell(title, "Use Up/Down keys and press Enter")
    choice = questionary.select(
        "Select an option",
        choices=[questionary.Choice(label, value=value) for label, value in options],
        use_shortcuts=False,
        qmark="",
        pointer=">",
        style=SELECT_STYLE,
    ).ask()
    return choice


def get_images_from_directory(directory):
    """Return sorted image file list from a directory."""
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def select_image_from_directory(directory, title):
    """Show arrow-key selector for available images in a directory."""
    images = get_images_from_directory(directory)
    if not images:
        clear_screen()
        console.print(f"[bold red]X No images found in {directory}[/bold red]")
        Prompt.ask("Press Enter to continue")
        return None

    options = [(img.name, str(img)) for img in images]
    return select_from_menu(title, options)


def display_menu():
    """Display main menu"""
    return select_from_menu(
        "Watermark Studio CLI",
        [
            ("Embed Watermark", "1"),
            ("Apply Attack", "2"),
            ("Extract Watermark", "3"),
            ("Evaluate Watermark", "4"),
            ("Exit", "5"),
        ],
    )


def select_input_image():
    """Let user select an input image"""
    image_path = select_image_from_directory(
        INPUT_IMAGES_DIR,
        "Choose Input Image",
    )
    if not image_path:
        return None
    
    if not Path(image_path).exists():
        console.print("[bold red]X Image file not found![/bold red]")
        return None
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            console.print("[bold red]X Failed to load image![/bold red]")
            return None
        console.print(f"[green]OK Image loaded successfully[/green] ({image.shape[1]}x{image.shape[0]})")
        return image_path
    except Exception as e:
        console.print(f"[bold red]X Error: {e}[/bold red]")
        return None


def select_watermark():
    """Let user select watermark type and source"""
    watermark_type = select_from_menu(
        "Choose Watermark Type",
        [
            ("Image watermark", "1"),
            ("Text watermark", "2"),
        ],
    )
    
    if watermark_type == "1":
        watermark_path = select_image_from_directory(
            WATERMARKS_DIR,
            "Choose Watermark Image",
        )
        if not watermark_path:
            return None
        return watermark_path
    else:
        clear_screen()
        console.print("[dim](Text watermark input)[/dim]")
        text = Prompt.ask("Enter watermark text")
        return text


def embed_watermark():
    """Embed watermark using Hybrid DWT-DCT algorithm."""
    clear_screen()
    render_shell("Embed Watermark", "Hybrid DWT-DCT Algorithm")

    image_path = select_input_image()
    if not image_path:
        return

    watermark = select_watermark()
    if not watermark:
        return

    # Build output path
    WATERMARKED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    stem   = Path(image_path).stem
    output = WATERMARKED_IMAGES_DIR / f"watermarked_{stem}.png"

    console.print("\n[cyan]Running DWT-DCT embedding…[/cyan]")
    try:
        result = run_embedding_pipeline(
            image_path      = image_path,
            watermark_input = watermark,
            output_path     = str(output),
            alpha           = ALPHA,
        )
        psnr_val = result["psnr"]
        n_bits   = result["n_bits"]
        capacity = result["capacity"]

        table = Table(title="Embedding Results", show_header=True)
        table.add_column("Metric",  style="cyan")
        table.add_column("Value",   style="green")
        table.add_row("Output file",     str(output))
        table.add_row("PSNR",            f"{psnr_val:.2f} dB  {'✓ PASS (>38 dB)' if psnr_val >= 38 else '✗ below target'}")
        table.add_row("Bits embedded",   f"{n_bits} / {capacity}")
        table.add_row("Alpha (strength)",f"{ALPHA}")
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]X Embedding failed: {e}[/bold red]")

    Prompt.ask("\nPress Enter to continue")


def apply_attack():
    """Apply attack method to image"""
    image_path = select_image_from_directory(
        WATERMARKED_IMAGES_DIR,
        "Choose Watermarked Image",
    )
    if not image_path:
        return
    
    image = cv2.imread(image_path)
    
    choice = select_from_menu(
        "Choose Attack Method",
        [(attack["name"], key) for key, attack in ATTACK_METHODS.items()],
    )
    selected_attack = ATTACK_METHODS[choice]
    
    # Apply attack with appropriate parameters
    console.print(f"\n[cyan]Applying {selected_attack['name']}...[/cyan]")
    
    try:
        if choice == "1":  # JPEG Compression
            quality = int(Prompt.ask("Quality (0-100)", default="75"))
            attacked_image = selected_attack["func"](image, quality=quality)
        elif choice == "2":  # Gaussian Noise
            sigma = int(Prompt.ask("Sigma (5-50)", default="10"))
            attacked_image = selected_attack["func"](image, sigma=sigma)
        elif choice == "3":  # Blurring
            kernel_size = int(Prompt.ask("Kernel size (odd number)", default="5"))
            if kernel_size % 2 == 0:
                kernel_size += 1
            attacked_image = selected_attack["func"](image, kernel_size=kernel_size)
        elif choice == "4":  # Cropping
            height, width = image.shape[:2]
            crop_mode = select_from_menu(
                "Choose Cropping Mode",
                [
                    ("Crop by percentage", "percentage"),
                    ("Crop specific region", "region"),
                ],
            )

            if crop_mode == "percentage":
                percentage = float(Prompt.ask("Crop percentage (0-0.5)", default="0.1"))
                attacked_image = selected_attack["func"](image, percentage=percentage)
            else:
                console.print(f"[dim]Image size: width={width}, height={height}[/dim]")
                x_start = int(Prompt.ask("x_start", default="0"))
                y_start = int(Prompt.ask("y_start", default="0"))
                x_end = int(Prompt.ask("x_end", default=str(width)))
                y_end = int(Prompt.ask("y_end", default=str(height)))

                if x_end <= x_start or y_end <= y_start:
                    raise ValueError("Invalid crop region: x_end must be > x_start and y_end must be > y_start")

                attacked_image = selected_attack["func"](
                    image,
                    crop_box=(x_start, y_start, x_end, y_end),
                )

                if attacked_image.size == 0:
                    raise ValueError("Crop region produced an empty image. Please choose a larger region.")
        elif choice == "5":  # Scaling
            scale = float(Prompt.ask("Scale factor", default="0.5"))
            attacked_image = selected_attack["func"](image, scale_factor=scale)
        
        # Save attacked image to results/attack_results; JPEG attack is exported as .jpg.
        ATTACK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        source_name = Path(image_path).stem
        attack_name = selected_attack["name"].lower().replace(" ", "_")
        output_extension = ".jpg" if choice == "1" else Path(image_path).suffix.lower()
        if not output_extension:
            output_extension = ".png"
        output_path = ATTACK_RESULTS_DIR / f"attacked_{attack_name}_{source_name}{output_extension}"
        cv2.imwrite(str(output_path), attacked_image)
        console.print(f"[green]OK Attack applied and saved to {output_path}[/green]\n")
    except Exception as e:
        console.print(f"[bold red]X Error applying attack: {e}[/bold red]\n")
    
    Prompt.ask("Press Enter to continue")


def extract_watermark_cli():
    """Extract watermark from a watermarked (possibly attacked) image."""
    clear_screen()
    render_shell("Extract Watermark", "Hybrid DWT-DCT Blind Extraction")

    # Choose extraction source first to avoid redundant prompts.
    attack_mode = select_from_menu(
        "Extraction Mode",
        [
            ("Extract from unattacked image", "clean"),
            ("Extract from attacked image", "attacked"),
        ],
    )

    if attack_mode == "attacked":
        image_path = select_image_from_directory(
            ATTACK_RESULTS_DIR,
            "Choose Attacked Image",
        )
    else:
        image_path = select_image_from_directory(
            WATERMARKED_IMAGES_DIR,
            "Choose Watermarked Image",
        )

    if not image_path:
        return

    # Get embedding parameters
    n_bits = int(Prompt.ask("Number of watermark bits", default="100"))
    wm_rows = int(Prompt.ask("Watermark grid rows", default="10"))
    wm_cols = int(Prompt.ask("Watermark grid cols", default="10"))
    alpha_val = float(Prompt.ask("Alpha (embedding strength)", default=str(ALPHA)))

    arnold_use = select_from_menu(
        "Arnold Descrambling",
        [
            ("No scrambling was used", "no"),
            ("Yes, descramble", "yes"),
        ],
    )

    arnold_iters = 0
    if arnold_use == "yes":
        arnold_iters = int(Prompt.ask("Arnold iterations", default="5"))

    # Build output path
    EXTRACTED_ATTACKED_WATERMARKS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    output = EXTRACTED_ATTACKED_WATERMARKS_DIR / f"extracted_{stem}.png"

    console.print("\n[cyan]Running watermark extraction…[/cyan]")
    try:
        watermark_shape = (wm_rows, wm_cols)
        result = run_extraction_pipeline(
            image_path=str(image_path),
            n_bits=n_bits,
            watermark_shape=watermark_shape,
            alpha=alpha_val,
            arnold_iterations=arnold_iters,
            output_path=str(output),
        )

        table = Table(title="Extraction Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Output file", str(output))
        table.add_row("Bits extracted", str(result["n_bits_extracted"]))
        table.add_row("Watermark shape", f"{watermark_shape}")
        table.add_row("Arnold descrambled", "Yes" if result["arnold_descrambled"] else "No")
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]X Extraction failed: {e}[/bold red]")

    Prompt.ask("\nPress Enter to continue")


def evaluate_watermark():
    """Evaluate watermark robustness metrics."""
    choice = select_from_menu(
        "Choose Evaluation Method",
        [(method, key) for key, method in EVALUATION_METHODS.items()],
    )
    
    console.print("\n[cyan]Computing metrics...[/cyan]")
    
    try:
        clear_screen()
        render_shell("Evaluate Watermark", "Computed robustness metrics")
        results_table = Table(title="Evaluation Results", show_header=True)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        if choice == "1":  # PSNR
            original_path = select_image_from_directory(
                INPUT_IMAGES_DIR,
                "Choose Original Image (assets/input_images)",
            )
            if not original_path:
                return

            watermarked_path = select_image_from_directory(
                WATERMARKED_IMAGES_DIR,
                "Choose Watermarked Image (assets/watermarked_images)",
            )
            if not watermarked_path:
                return

            original = cv2.imread(original_path)
            watermarked = cv2.imread(watermarked_path)
            psnr_val = calculate_psnr(original, watermarked)
            results_table.add_row("PSNR", f"{psnr_val:.2f} dB")
            if psnr_val >= 40:
                psnr_note = "Excellent - watermark is visually imperceptible"
            elif psnr_val >= 35:
                psnr_note = "Good - minor perceptual difference"
            elif psnr_val >= 30:
                psnr_note = "Fair - visible distortion may appear"
            else:
                psnr_note = "Poor - significant quality degradation"
            results_table.add_row("Interpretation", psnr_note)
        elif choice == "2":  # SSIM
            original_path = select_image_from_directory(
                INPUT_IMAGES_DIR,
                "Choose Original Image (assets/input_images)",
            )
            if not original_path:
                return

            watermarked_path = select_image_from_directory(
                WATERMARKED_IMAGES_DIR,
                "Choose Watermarked Image (assets/watermarked_images)",
            )
            if not watermarked_path:
                return

            original = cv2.imread(original_path)
            watermarked = cv2.imread(watermarked_path)
            ssim_val = calculate_ssim(original, watermarked)
            results_table.add_row("SSIM", f"{ssim_val:.4f}")
            if ssim_val >= 0.98:
                ssim_note = "Excellent - near-identical structure"
            elif ssim_val >= 0.95:
                ssim_note = "Good - high structural similarity"
            elif ssim_val >= 0.90:
                ssim_note = "Fair - moderate structural change"
            else:
                ssim_note = "Poor - notable structural distortion"
            results_table.add_row("Interpretation", ssim_note)
        elif choice == "3":  # BER
            original_watermark_path = select_image_from_directory(
                WATERMARKS_DIR,
                "Choose Original Watermark (assets/watermarks)",
            )
            if not original_watermark_path:
                return

            extracted_watermark_path = select_image_from_directory(
                EXTRACTED_ATTACKED_WATERMARKS_DIR,
                "Choose Extracted Attacked Watermark (results/extracted_attacked_watermarks)",
            )
            if not extracted_watermark_path:
                return

            original_watermark = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
            extracted_watermark = cv2.imread(extracted_watermark_path, cv2.IMREAD_GRAYSCALE)
            if original_watermark is None or extracted_watermark is None:
                raise ValueError("Failed to read watermark image(s). Please verify the selected files.")

            ber_val = calculate_ber(original_watermark, extracted_watermark)
            results_table.add_row("BER", f"{ber_val:.6f}")
            if ber_val <= 0.01:
                ber_note = "Excellent - very reliable extraction"
            elif ber_val <= 0.05:
                ber_note = "Good - robust with minor bit errors"
            elif ber_val <= 0.10:
                ber_note = "Fair - extraction is somewhat degraded"
            else:
                ber_note = "Poor - extraction quality is weak"
            results_table.add_row("Interpretation", ber_note)
        
        console.print(results_table)
    except Exception as e:
        console.print(f"[bold red]X Error during evaluation: {e}[/bold red]")
    
    console.print()
    Prompt.ask("Press Enter to continue")


def main():
    """Main CLI loop"""
    while True:
        choice = display_menu()

        if choice == "1":
            embed_watermark()
        elif choice == "2":
            apply_attack()
        elif choice == "3":
            extract_watermark_cli()
        elif choice == "4":
            evaluate_watermark()
        elif choice == "5":
            console.print("\n[cyan]Goodbye![/cyan]\n")
            break


if __name__ == "__main__":
    main()