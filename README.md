
# ChildFacePrivacyGAN

🔐 Privacy-conscious synthetic face generation platform for child faces with ethical AI safeguards.  
Developed using Streamlit and DeepFace.

---

## 🚀 Features

- Upload a child face image (with ethical consent).
- Generate a synthetic, anonymized version with visual variation.
- Compare face embeddings using cosine similarity.
- Graphical similarity score display (bar chart).
- DeepFace (Facenet) integration for robust representation.
- Modular and ready for future GAN + ArcFace extensions.

---

## ▶️ How to Run (Locally or via Streamlit Cloud)

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ChildFacePrivacyGAN.git
cd ChildFacePrivacyGAN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

---

## 📦 requirements.txt (Dependencies)

```
streamlit
Pillow
numpy
matplotlib
scikit-learn
opencv-python-headless
deepface
tensorflow
```

---

## ⚠️ Ethical Disclaimer

This tool is designed **strictly for academic and ethical research use**.  
- It must not be used for clinical diagnosis or any unauthorized data processing.  
- Use only with proper ethics board approval and informed guardian consent.  
- Generated faces do not represent real identities.

---

© 2025 ChildFacePrivacyGAN. All rights reserved.
