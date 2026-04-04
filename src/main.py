"""
Image Watermarking CLI
Interactive command-line interface for watermarking, attacks, and evaluation
"""

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from pathlib import Path
import cv2
import os
import questionary

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

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_IMAGES_DIR = PROJECT_ROOT / "assets" / "input_images"
WATERMARKS_DIR = PROJECT_ROOT / "assets" / "watermarks"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

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
    "4": "Full Robustness Evaluation",
}


def clear_screen():
    """Clear terminal screen between menus on Windows."""
    os.system("cls")


def select_from_menu(title, options):
    """Arrow-key menu selection using up/down and Enter."""
    clear_screen()
    console.print(title)
    choice = questionary.select(
        "Use ↑/↓ and Enter:",
        choices=[questionary.Choice(label, value=value) for label, value in options],
        use_shortcuts=False,
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
        "\n[bold cyan]Image Watermarking Framework CLI[/bold cyan]\n",
        [
            ("Embed Watermark", "1"),
            ("Apply Attack", "2"),
            ("Evaluate Image Quality", "3"),
            ("Exit", "4"),
        ],
    )


def select_input_image():
    """Let user select an input image"""
    image_path = select_image_from_directory(
        INPUT_IMAGES_DIR,
        "\n[bold]Input Image Selection[/bold]\n",
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
        "\n[bold]Watermark Selection[/bold]\n",
        [
            ("Image watermark", "1"),
            ("Text watermark", "2"),
        ],
    )
    
    if watermark_type == "1":
        watermark_path = select_image_from_directory(
            WATERMARKS_DIR,
            "\n[bold]Watermark Image Selection[/bold]\n",
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
    """Embed watermark (not implemented yet)"""
    clear_screen()
    console.print("\n[bold yellow]>> Watermark embedding is not yet implemented[/bold yellow]")
    console.print("[dim]This feature will be available in a future update.[/dim]\n")
    
    image_path = select_input_image()
    if not image_path:
        return
    
    watermark = select_watermark()
    if not watermark:
        return
    
    console.print("\n[cyan]Watermark embedding configuration:[/cyan]")
    console.print(f"  Image: {image_path}")
    console.print(f"  Watermark: {watermark}")
    console.print("\n[dim]Coming soon...[/dim]\n")
    Prompt.ask("Press Enter to continue")


def apply_attack():
    """Apply attack method to image"""
    image_path = select_input_image()
    if not image_path:
        return
    
    image = cv2.imread(image_path)
    
    choice = select_from_menu(
        "\n[bold]Select Attack Method[/bold]\n",
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
            percentage = float(Prompt.ask("Crop percentage (0-1)", default="0.1"))
            attacked_image = selected_attack["func"](image, percentage=percentage)
        elif choice == "5":  # Scaling
            scale = float(Prompt.ask("Scale factor", default="0.5"))
            attacked_image = selected_attack["func"](image, scale_factor=scale)
        
        # Save attacked image
        output_path = f"attacked_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, attacked_image)
        console.print(f"[green]OK Attack applied and saved to {output_path}[/green]\n")
    except Exception as e:
        console.print(f"[bold red]X Error applying attack: {e}[/bold red]\n")
    
    Prompt.ask("Press Enter to continue")


def evaluate_quality():
    """Evaluate image quality"""
    original_path = select_image_from_directory(
        INPUT_IMAGES_DIR,
        "\n[bold]Select Original Image[/bold]\n",
    )
    if not original_path:
        return

    attacked_path = select_image_from_directory(
        INPUT_IMAGES_DIR,
        "\n[bold]Select Attacked/Degraded Image[/bold]\n",
    )
    if not attacked_path:
        return
    
    original = cv2.imread(original_path)
    attacked = cv2.imread(attacked_path)
    
    choice = select_from_menu(
        "\n[bold]Select Evaluation Method[/bold]\n",
        [(method, key) for key, method in EVALUATION_METHODS.items()],
    )
    
    console.print("\n[cyan]Computing metrics...[/cyan]")
    
    try:
        results_table = Table(title="Evaluation Results", show_header=True)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        if choice == "1":  # PSNR
            psnr_val = calculate_psnr(original, attacked)
            results_table.add_row("PSNR", f"{psnr_val:.2f} dB")
        elif choice == "2":  # SSIM
            ssim_val = calculate_ssim(original, attacked)
            results_table.add_row("SSIM", f"{ssim_val:.4f}")
        elif choice == "3":  # BER
            console.print("[dim]Note: BER requires extracted watermark (not available)[/dim]")
        elif choice == "4":  # Full robustness
            psnr_val = calculate_psnr(original, attacked)
            ssim_val = calculate_ssim(original, attacked)
            results_table.add_row("PSNR", f"{psnr_val:.2f} dB")
            results_table.add_row("SSIM", f"{ssim_val:.4f}")
        
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
            evaluate_quality()
        elif choice == "4":
            console.print("\n[cyan]Goodbye![/cyan]\n")
            break


if __name__ == "__main__":
    main()
