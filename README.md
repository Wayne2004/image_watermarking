# 🔐 Robust Image Watermarking using DCT, DWT & Hybrid Techniques

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

---

## 📌 Overview

Digital watermarking is a crucial technique for protecting **ownership, authenticity, and integrity of digital media**.

This project implements classical watermarking techniques:
- Discrete Cosine Transform (DCT)
- Discrete Wavelet Transform (DWT)
- Hybrid DCT-DWT

It further enhances these techniques with **adaptive embedding strategies, robustness evaluation, and real-world attack simulations**.

---

## 🎯 Objectives

- Implement and compare DCT, DWT, and Hybrid watermarking techniques  
- Improve robustness against real-world image distortions  
- Maintain high imperceptibility (minimal visual degradation)  
- Provide quantitative evaluation using standard metrics  

---

## 🚀 Key Features

- ✅ DCT-based watermark embedding & extraction  
- ✅ DWT-based watermark embedding & extraction  
- ✅ Hybrid DCT-DWT watermarking  
- ✅ Adaptive embedding strategy *(enhancement)*  
- ✅ Attack simulation framework  
- ✅ Automated evaluation metrics (PSNR, SSIM, BER)  
- ✅ Visualization of results  

---

## 🧠 Contributions

This project goes beyond basic implementation by introducing:

### 🔍 Adaptive Embedding
Dynamically selects optimal embedding regions based on image characteristics to improve invisibility and robustness.

### 🛡️ Attack Simulation Framework
Simulates real-world distortions:
- JPEG compression  
- Gaussian noise  
- Blurring / filtering  
- Cropping  

### 📊 Automated Evaluation
Measures watermark performance using:
- **PSNR** – image quality  
- **SSIM** – structural similarity  
- **BER** – extraction accuracy  

### ⚙️ Hybrid Optimization
Combines DCT and DWT to achieve a better balance between robustness and imperceptibility.

---

## 🏗️ Project Structure
```bash
image_watermarking/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── attacks/
│   │   ├── __init__.py
│   │   └── attacks.py
│   ├── dct/
│   ├── dwt/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluation.py
│   └── hybrid/
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
│
├── examples.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```


---

## ⚙️ Installation

```bash
git clone https://github.com/Wayne2004/image_watermarking.git
cd image_watermarking

pip install -r requirements.txt
```

## ⚡ Execution
```bash
python src/main.py
```
