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
/* Quitar los clips que aparecen junto a los títulos */
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
    display: none !important;
}

/* Ajustar márgenes para que no queden espacios en blanco */
h1, h2, h3, h4, h5, h6 {
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}
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
    st.session_state.session_models = {}  # modelos subidos en la sesión

# ---------- Sidebar: modelos ----------
st.sidebar.header("⚙️ Modelos")

uploaded_model_files = st.sidebar.file_uploader(
    "Sube uno o varios modelos (.h5)", 
    type=['h5'], 
    accept_multiple_files=True,
    key="upload_modelos"
)

# Procesar archivos subidos
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

# Modelos preentrenados en carpeta
folder_models = {}
if os.path.exists(MODEL_DIR):
    for f in os.listdir(MODEL_DIR):
        if f.endswith('.h5'):
            folder_models[f] = os.path.join(MODEL_DIR, f)
else:
    os.makedirs(MODEL_DIR)

# Opciones combinadas
select_options = [f"[Preentrenado] {n}" for n in folder_models.keys()] + \
                 [f"[Subido] {n}" for n in st.session_state.session_models.keys()]

default_index = 0
if "[Preentrenado] modelo_gatos_perros_mobilenet.h5" in select_options:
    default_index = select_options.index("[Preentrenado] modelo_gatos_perros_mobilenet.h5")

selected_model_label = st.sidebar.selectbox(
    "Selecciona un modelo",
    select_options,
    index=default_index if select_options else 0
)

# Cargar modelo seleccionado
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
        st.sidebar.success(f"✅ Modelo principal {selected_model_name} cargado")
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar modelo: {e}")

st.write("---")

# ---------- Subir múltiples imágenes ----------
st.markdown("📂 Sube una o varias imágenes (JPG, PNG)")
uploaded_images = st.file_uploader(
    "",  # dejamos el label vacío
    type=['jpg','jpeg','png'],
    accept_multiple_files=True,
    key="upload_imagenes"
)
if uploaded_images:
    for uploaded_image in uploaded_images:
        try:
            image_pil = Image.open(uploaded_image)

            # Nombre de la imagen arriba
            st.subheader(uploaded_image.name)

            # Imagen centrada
            st.image(image_pil, use_container_width=True)

            if model is None:
                st.warning('⚠️ Aún no hay un modelo cargado.')
            else:
                img = image_pil.convert('RGB').resize(IMG_SIZE)
                img_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0)

                with st.spinner(f'🔎 Clasificando {uploaded_image.name}...'):
                    preds = model.predict(img_array)

                # Mostrar resultados
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
