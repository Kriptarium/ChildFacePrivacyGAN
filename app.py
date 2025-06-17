
import streamlit as st
from PIL import Image
import numpy as np
import io
import time
import insightface
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

st.title("ChildFacePrivacyGAN - InsightFace ile Yüz Benzerliği")

uploaded_file = st.file_uploader("Bir çocuk yüzü fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    model = insightface.app.FaceAnalysis(name='buffalo_l')
    model.prepare(ctx_id=-1)
    return model

model = load_model()

def get_embedding(image):
    np_img = np.array(image)
    faces = model.get(np_img)
    if not faces:
        return None
    return faces[0].embedding

def create_synthetic_face(image):
    # Basit sentetik dönüşüm (renk azaltma ve flip)
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT).convert("L").convert("RGB")
    return flipped

def plot_similarity(score):
    fig, ax = plt.subplots()
    ax.barh(['Benzerlik Skoru'], [score], color='green')
    ax.set_xlim([0, 1])
    ax.set_xlabel('0 (farklı) - 1 (benzer)')
    st.pyplot(fig)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    if st.button("Sentetik Yüz Üret ve Karşılaştır"):
        with st.spinner("İşleniyor..."):
            time.sleep(2)
            synthetic = create_synthetic_face(image)
            st.image(synthetic, caption="Sentetik Yüz", use_column_width=True)

            emb1 = get_embedding(image)
            emb2 = get_embedding(synthetic)

            if emb1 is not None and emb2 is not None:
                score = cosine_similarity([emb1], [emb2])[0][0]
                st.markdown(f"### 🔍 Yüz Benzerlik Skoru: `{score:.4f}`")
                plot_similarity(score)
                if score < 0.5:
                    st.success("✅ Sentetik yüz anonimlik açısından yeterince farklı.")
                else:
                    st.warning("⚠️ Sentetik yüz orijinalle yüksek benzerlik taşıyor.")
            else:
                st.error("Yüz algılanamadı. Lütfen daha net bir fotoğraf yükleyiniz.")

        buf = io.BytesIO()
        synthetic.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("Sentetik Yüzü İndir", data=byte_im, file_name="sentetik_yuz.png", mime="image/png")

st.markdown("---")
st.markdown("⚠️ Bu uygulama yalnızca etik izinli araştırmalar içindir. Klinik tanı amacıyla kullanılmaz.")
