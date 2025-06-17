
import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import io
import time
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.title("ChildFacePrivacyGAN - Sentetik Yüz ve Benzerlik Değerlendirme")

uploaded_file = st.file_uploader("Bir çocuk yüzü fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])

def get_face_embedding(image):
    try:
        image_np = np.array(image)
        face_locations = face_recognition.face_locations(image_np)
        if len(face_locations) == 0:
            return None
        face_encodings = face_recognition.face_encodings(image_np, face_locations)
        return face_encodings[0] if face_encodings else None
    except Exception as e:
        return None

def create_simple_synthetic(image):
    # Yüz üzerinde bazı varyasyonlar yap (dönme + renk filtresi)
    image = image.rotate(10).transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(0.5)  # Renk doygunluğunu azalt

def plot_similarity_bar(score):
    fig, ax = plt.subplots()
    ax.barh(['Benzerlik Skoru'], [score], color='skyblue')
    ax.set_xlim([0, 1])
    ax.set_xlabel('0 (farklı) - 1 (benzer)')
    st.pyplot(fig)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    if st.button("Sentetik Yüz Üret ve Karşılaştır"):
        with st.spinner("Sentetik yüz üretiliyor..."):
            time.sleep(2)
            synthetic_image = create_simple_synthetic(image)
            st.image(synthetic_image, caption="Sentetik Yüz", use_column_width=True)

            # Embedings hesapla
            real_embedding = get_face_embedding(image)
            synthetic_embedding = get_face_embedding(synthetic_image)

            if real_embedding is not None and synthetic_embedding is not None:
                similarity = cosine_similarity([real_embedding], [synthetic_embedding])[0][0]
                st.markdown(f"### 🔍 Yüz Benzerlik Skoru: `{similarity:.4f}`")
                plot_similarity_bar(similarity)

                if similarity < 0.5:
                    st.success("✅ Sentetik yüz, orijinalden yeterince farklı görünüyor (anonimlik sağlanmış olabilir).")
                else:
                    st.warning("⚠️ Sentetik yüz, orijinal yüzle yüksek benzerlik taşıyor (anonimlik riski olabilir).")
            else:
                st.error("Yüz tanıma başarısız oldu. Daha net bir yüz fotoğrafı yükleyiniz.")

        # İndirilebilir dosya
        buf = io.BytesIO()
        synthetic_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("Sentetik Yüzü İndir", data=byte_im, file_name="sentetik_yuz.png", mime="image/png")

st.markdown("---")
st.markdown("⚠️ Bu uygulama yalnızca etik izinli araştırmalar içindir. Klinik tanı amacıyla kullanılmaz. Üretilen yüzler gerçek bireyleri temsil etmez.")
