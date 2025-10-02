import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import tempfile
import time
import shutil

# ---------- Configuración de página ----------
st.set_page_config(page_title="Clasificador Gato/Perro", layout="centered", page_icon="🐾")

# ---------- Quitar clips automáticos de los títulos ----------
st.markdown("""
<style>
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }
h1, h2, h3, h4, h5, h6 { margin-top: 0.5rem; margin-bottom: 0.5rem; }
.centered-image img { max-width: 100%; height: auto; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.6); }
.centered-image { text-align: center; margin-bottom: 1rem; }
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
    "Sube uno o varios modelos (.h5)", 
    type=['h5'], 
    accept_multiple_files=True,
    key="upload_modelos"
)

# Procesar modelos subidos
if uploaded_model_files:
    for file in uploaded_model_files:
        if file.name not in st.session_state.session_models:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
                    tmp_file.write(file.read())
                    tmp_path = tmp_file.name
                st.session_state.session_models[file.name] = tmp_path
                msg_container = st.sidebar.empty()
                msg_container.success(f"✅ Modelo {file.name} cargado")
                time.sleep(2)
                msg_container.empty()
            except Exception as e:
                st.sidebar.error(f"❌ Error al subir {file.name}: {e}")

# ---------- Actualizar modelos antiguos ----------
folder_models = {}
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

for f in os.listdir(MODEL_DIR):
    if f.endswith(".h5"):
        old_path = os.path.join(MODEL_DIR, f)
        try:
            # Intentar cargar y guardar para TF 2.12
            model_tmp = tf.keras.models.load_model(old_path)
            new_name = f.replace(".h5", "_tf12.h5")
            new_path = os.path.join(MODEL_DIR, new_name)
            model_tmp.save(new_path)
            folder_models[new_name] = new_path
        except Exception:
            # Si no se puede actualizar, se usa el original
            folder_models[f] = old_path

# Combinar con modelos subidos
select_options = [f"[Preentrenado] {n}" for n in folder_models.keys()] + \
                 [f"[Subido] {n}" for n in st.session_state.session_models.keys()]

default_index = 0
if "[Preentrenado] modelo_gatos_perros_mobilenet_tf12.h5" in select_options:
    default_index = select_options.index("[Preentrenado] modelo_gatos_perros_mobilenet_tf12.h5")

selected_model_label = st.sidebar.selectbox(
    "Selecciona un modelo",
    select_options,
    index=default_index if select_options else 0
)

# ---------- Cargar modelo seleccionado ----------
model = None
if select_options:
    try:
        if selected_model_label.startswith("[Preentrenado]"):
            selected_model_name = selected_model_label.replace("[Preentrenado] ", "")
            model_path = folder_models[selected_model_name]
        else:
            selected_model_name = selected_model_label.replace("[Subido] ", "")
            model_path = st.session_state.session_models[selected_model_name]

        with st.spinner(f"Cargando modelo {selected_model_name}..."):
            model = tf.keras.models.load_model(model_path)
        st.sidebar.success(f"✅ Modelo {selected_model_name} cargado")
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar modelo: {e}")

st.write("---")

# ---------- Subir múltiples imágenes ----------
st.markdown("📂 Sube una o varias imágenes (JPG, PNG)")
uploaded_images = st.file_uploader(
    "",  
    type=['jpg','jpeg','png'],
    accept_multiple_files=True,
    key="upload_imagenes"
)

if uploaded_images:
    for uploaded_image in uploaded_images:
        try:
            image_pil = Image.open(uploaded_image)
            st.subheader(f"📷 {uploaded_image.name}")
            st.markdown("<div class='centered-image'>", unsafe_allow_html=True)
            st.image(image_pil, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if model is None:
                st.warning('⚠️ Aún no hay un modelo cargado.')
            else:
                img = image_pil.convert('RGB').resize(IMG_SIZE)
                img_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0)

                with st.spinner(f'🔎 Clasificando {uploaded_image.name}...'):
                    preds = model.predict(img_array)

                if preds.ndim == 2 and preds.shape[1] > 1:
                    probs = preds[0]
                    idx = np.argmax(probs)
                    label = CLASSES[idx] if idx < len(CLASSES) else f'Clase {idx}'
                    confidence = float(probs[idx])
                else:
                    val = float(preds.ravel()[0])
                    label = CLASSES[1] if val >= 0.5 else CLASSES[0]
                    confidence = val if val >= 0.5 else 1 - val

                st.success(f'✅ Predicción: **{label}** (confianza: {confidence*100:.2f}%)')

        except Exception as e:
            st.error(f'❌ No se pudo procesar {uploaded_image.name}: {e}')
else:
    st.info('ℹ️ Sube una o varias imágenes para ver las predicciones.')

st.write('---')
