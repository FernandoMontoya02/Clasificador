import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import tempfile
import time

# ---------- Configuración de página ----------
st.set_page_config(page_title="Clasificador Gato/Perro", layout="centered", page_icon="🐾")

# ---------- Quitar clips automáticos de los títulos ----------
st.markdown("""
<style>
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }
h1, h2, h3, h4, h5, h6 { margin-top:0.5rem; margin-bottom:0.5rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Título ----------
st.title("Clasificador de Gatos y Perros")
st.write("Sube imágenes y modelos para probar tus predicciones.")
st.write("---")

# ---------- Configuración ----------
IMG_SIZE = (224, 224)
CLASSES = ['gatos', 'perros']
MODEL_DIR = "modelos"

if 'session_models' not in st.session_state:
    st.session_state.session_models = {}

# ---------- Sidebar: modelos ----------
st.sidebar.header("⚙️ Modelos")
uploaded_model_files = st.sidebar.file_uploader(
    "Sube modelos (.h5)", type=['h5'], accept_multiple_files=True
)

# Guardar modelos subidos temporalmente
if uploaded_model_files:
    for file in uploaded_model_files:
        if "_tf2" in file.name:
            continue  # ignorar modelos tf2
        if file.name not in st.session_state.session_models:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
                tmp_file.write(file.read())
                st.session_state.session_models[file.name] = tmp_file.name
            st.sidebar.success(f"✅ Modelo {file.name} cargado")

# Modelos preentrenados en la carpeta, ignorando los *_tf2
folder_models = {}
if os.path.exists(MODEL_DIR):
    for f in os.listdir(MODEL_DIR):
        if f.endswith('.h5') and "_tf2" not in f:
            folder_models[f] = os.path.join(MODEL_DIR, f)
else:
    os.makedirs(MODEL_DIR)

# Combinar opciones
select_options = [f"[Preentrenado] {n}" for n in folder_models.keys()] + \
                 [f"[Subido] {n}" for n in st.session_state.session_models.keys()]

selected_model_label = st.sidebar.selectbox("Selecciona un modelo", select_options)

# Cargar modelo
model = None
if selected_model_label:
    try:
        if selected_model_label.startswith("[Preentrenado]"):
            model_path = folder_models[selected_model_label.replace("[Preentrenado] ","")]
        else:
            model_path = st.session_state.session_models[selected_model_label.replace("[Subido] ","")]

        model = tf.keras.models.load_model(model_path)
        st.sidebar.success(f"✅ Modelo cargado: {selected_model_label}")
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar modelo: {e}")

st.write("---")

# ---------- Subir imágenes ----------
st.markdown("📂 Sube una o varias imágenes (JPG, PNG)")
uploaded_images = st.file_uploader("", type=['jpg','jpeg','png'], accept_multiple_files=True)

if uploaded_images:
    for uploaded_image in uploaded_images:
        try:
            image_pil = Image.open(uploaded_image)
            st.subheader(uploaded_image.name)
            st.image(image_pil, use_container_width=True)

            if model:
                img = image_pil.convert('RGB').resize(IMG_SIZE)
                img_array = np.expand_dims(np.array(img)/255.0, axis=0)

                preds = model.predict(img_array)
                if preds.ndim==2 and preds.shape[1]>1:
                    idx = np.argmax(preds[0])
                    confidence = float(preds[0][idx])
                    label = CLASSES[idx]
                else:
                    val = float(preds.ravel()[0])
                    label = CLASSES[1] if val>=0.5 else CLASSES[0]
                    confidence = val if val>=0.5 else 1-val

                st.success(f"✅ Predicción: **{label}** (confianza: {confidence*100:.2f}%)")
            else:
                st.warning("⚠️ Aún no hay un modelo cargado.")
        except Exception as e:
            st.error(f"❌ No se pudo procesar {uploaded_image.name}: {e}")
else:
    st.info("ℹ️ Sube una o varias imágenes para ver las predicciones.")

st.write("---")
