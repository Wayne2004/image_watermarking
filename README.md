# Robust Image Watermarking (Hybrid DWT-DCT)
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/02/V1-tar_umt_logo_%28full_colour%29.png" width="120"/>
</p>

<p align="center">
  <b>A research-oriented implementation of an "Unbreakable" Blind Watermarking System</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Type-Research--Oriented-0969DA"/>
  <img src="https://img.shields.io/badge/Status-Optimized--Role--C-F59E0B"/>
</p>

This project implements a high-robustness blind image watermarking system using a Hybrid DWT-DCT approach. It is specifically engineered to survive severe geometric attacks (Cropping, Scaling) and signal processing attacks (JPEG, Noise).

## Core Breakthroughs (Role C Optimization)

Unlike standard implementations, this system features **Geometric Resynchronization**:

- **Virtual-Width Ribbon Synchronization**: Uses the host image's original dimensions as a geometric anchor to "re-stitch" watermarks after cropping.
- **Ultra-Lighthouse Header**: Employs a 4.0x Alpha-boosted 64-bit calibration header for unmistakable grid alignment.
- **Polarity-Aware Extraction**: Automatically detects and corrects bit-inversion (color flipping) caused by DWT phase shifts.
- **High-Redundancy Tiling**: Automatically tiles the watermark ribbon across the sub-bands to ensure multiple data recovery points.

## Current Features

- **Interactive CLI**: Full keyboard-driven experiment workflow (Embed -> Attack -> Extract -> Evaluate).
- **Hybrid DWT-DCT**: Embeds in frequency detail sub-bands for high invisibility and robustness.
- **Adaptive Strength**: HVS-inspired block-wise Alpha calculation.
- **Attack Suite**:
  - JPEG Compression (Quality 10-90)
  - Gaussian & Salt-and-Pepper Noise
  - Median Blurring
  - **Severe Cropping (10%-50% Area Removal)**
  - Scaling (Downsampling to 0.5x)
- **Advanced Metrics**:
  - **PSNR & SSIM**: Human-visual quality assessment.
  - **BER (Bit Error Rate)**: Quantitative data accuracy.
  - **NCC (Normalized Cross-Correlation)**: Centered binary correlation for structural verification.

## Project Structure

```text
image_watermarking/
├── src/
│   ├── main.py                # Main Entry Point (Interactive CLI)
│   ├── watermarking/
│   │   ├── watermarking.py    # Embedding & DWT-DCT Engine
│   │   ├── extraction.py      # Robust Resync Extractor
│   │   └── arnold.py          # Scrambling Security
│   ├── attacks/               # Attack Simulation Suite
│   └── evaluation/            # PSNR, BER, NCC Metrics
├── assets/
│   ├── input_images/          # Original Host Images
│   └── watermarks/            # Input Logo/Text & Baseline Reference
├── results/
│   ├── attack_results/        # Post-attack images
│   └── extracted_attacked_watermarks/ # Final extracted outputs
└── test/                      # Debug & Validation Scripts
```

## Installation (using uv)

```bash
git clone https://github.com/Wayne2004/image_watermarking.git
cd image_watermarking
# Create venv and install
uv venv
source .venv/bin/activate  # Linux
uv pip install -r requirements.txt
```

## Run

Interactive CLI:
```bash
uv run python src/main.py
```

Scientific Robustness Test:
```bash
uv run python test/debug_cropping.py
```

## Dependencies

- **numpy**: Matrix operations
- **opencv-python**: Image processing
- **Pillow**: Text rendering
- **PyWavelets**: Discrete Wavelet Transform
- **scipy**: Discrete Cosine Transform (DCT)
- **rich**: Professional CLI UI
- **questionary**: Interactive Prompts
- **pandas / seaborn / matplotlib**: Data analysis & Reporting

## Performance Targets

| Attack Type | Target BER | Target NCC |
| :--- | :--- | :--- |
| JPEG Quality 50 | < 0.05 | > 0.95 |
| 10% Crop | < 0.20 | > 0.85 |
| 0.5x Scaling | < 0.10 | > 0.90 |
| Clean | 0.00 | 1.00 |
