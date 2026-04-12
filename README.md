# Robust Image Watermarking (Hybrid DWT-DCT)
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/02/V1-tar_umt_logo_%28full_colour%29.png" width="120"/>
</p>

<p align="center">
  <b>A research-oriented project focused on designing and evaluating robust digital image watermarking techniques</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Type-Research--Oriented-0969DA"/>
  <img src="https://img.shields.io/badge/Status-In%20Progress-F59E0B"/>
</p>

Research-oriented implementation of a blind image watermarking system with:

- Hybrid DWT-DCT embedding
- Adaptive per-block embedding strength (HVS-inspired)
- Dual sub-band redundancy (LH + HL)
- Attack simulation and robustness evaluation
- Interactive CLI workflow for end-to-end experiments

## Overview

This repository currently focuses on a practical hybrid watermarking pipeline:

1. Embed watermark bits in DWT detail sub-bands after block-DCT.
2. Apply attacks to test robustness.
3. Extract watermark from clean or attacked outputs.
4. Evaluate quality and robustness using PSNR, SSIM, and BER.

The implementation supports image watermark inputs and text watermark input (text is rendered into a grayscale watermark image internally).

## Current Features

- Interactive CLI with keyboard-driven menus (questionary + rich)
- Hybrid DWT-DCT blind embedding and extraction
- Adaptive alpha (block-wise embedding strength)
- Dual sub-band embedding and extraction voting (LH + HL)
- Optional Arnold descrambling during extraction
- Attack suite:
  - JPEG compression
  - Gaussian noise
  - Blurring
  - Cropping (percentage or region)
  - Scaling
- Evaluation metrics:
  - PSNR
  - SSIM
  - BER
- In-CLI metric interpretation labels (excellent/good/fair/poor)

## Project Structure

```text
image_watermarking/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── demo_embedding.py
│   ├── demo_extraction.py
│   ├── attacks/
│   │   ├── __init__.py
│   │   └── attacks.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluation.py
│   └── watermarking/
│       ├── __init__.py
│       ├── watermarking.py
│       ├── extraction.py
│       └── arnold.py
│
├── assets/
│   ├── input_images/
│   ├── watermarked_images/
│   └── watermarks/
│
├── results/
│   ├── attack_results/
│   └── extracted_attacked_watermarks/
│
├── notebooks/
├── examples.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Installation

```bash
git clone https://github.com/Wayne2004/image_watermarking.git
cd image_watermarking
pip install -r requirements.txt
```

## Run

Interactive CLI:

```bash
python src/main.py
```

Optional demos:

```bash
python src/demo_embedding.py
python src/demo_extraction.py
python examples.py
```

## CLI Workflow

From the main menu:

1. Embed Watermark
2. Apply Attack
3. Extract Watermark
4. Evaluate Watermark

Input/output directories used by the CLI:

- Input host images: assets/input_images
- Watermark images: assets/watermarks
- Watermarked outputs: assets/watermarked_images
- Attacked outputs: results/attack_results
- Extracted watermark outputs: results/extracted_attacked_watermarks

## Notes

- Embedding currently aligns image dimensions for DWT/block processing; this may change effective processed dimensions for some inputs.
- Evaluation now aligns quality metrics without interpolation artifacts and uses nearest-neighbor alignment for BER resizing.

## Dependencies

- numpy
- opencv-python
- Pillow
- scikit-image
- rich-cli
- questionary
- PyWavelets
