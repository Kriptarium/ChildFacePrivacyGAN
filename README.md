
# ChildFacePrivacyGAN

**ChildFacePrivacyGAN** is a privacy-conscious Streamlit application designed to generate synthetic child face images from real input photos for research purposes. It emphasizes ethical data handling and is intended for use only with proper ethics approval.

---

## 📌 Project Description

This application allows users to upload a real child face image (with ethical clearance) and generates a synthetic version using reversible-safe transformations or, in full versions, a GAN-based generator. It is designed to ensure that no personally identifiable information is retained or reproduced.

---

## ⚙️ Installation

Make sure you have Python 3.8+ installed. Then install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

To run the application locally:

```bash
streamlit run app.py
```

1. Upload a child's face image (JPG or PNG).
2. Click **"Sentetik Yüz Üret"** to create a synthetic version.
3. Download the generated face for research or augmentation use.

---

## 🧠 Technical Notes

- Current version uses a mirrored transformation for demonstration.
- Future versions may integrate **StyleGAN2-ADA** or other privacy-aware GANs.
- The app processes images locally and does not store or transmit them.

---

## ⚠️ Ethical Disclaimer

This application is strictly for **academic research and educational purposes** under approved ethical frameworks.

- No real identity is reconstructed.
- Do not use this application for clinical diagnosis or unauthorized data processing.
- Always obtain **informed consent** from guardians before using child data.

---

© 2025 ChildFacePrivacyGAN Project. All rights reserved.
