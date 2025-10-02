import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import tempfile
import time

# ---------- Configuración de página ----------
st.set_page_config(page_title="Clasificador Gato/Perro", layout="centered", page_icon="🐾")

# ---------- Estilos y clips ----------
st.markdown("""
<style>
/* Quitar clips de títulos */
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }
h1, h2, h3, h4, h5, h6 { margin-top: 0.5rem; margin-bottom: 0.5rem; }

/* Imagen centrada */
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
    "Sube uno o varios modelos (.h5)", type=['h5'], accept_multiple_files=True
)

if uploaded_model_files:
    for file in uploaded_model_files:
        if file.name not in st.session_state.session_models:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
            tmp_file.write(file.read())
            tmp_file.close()
            st.session_state.session_models[file.name] = tmp_file.name
            st.sidebar.success(f"✅ Modelo {file.name} cargado")

# ---------- Función para cargar modelos antiguos ----------
def load_model_safe(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        # Detectar error de batch_shape
        if "Unrecognized keyword arguments: ['batch_shape']" in str(e):
            st.warning(f"⚠️ Modelo antiguo detectado, actualizando a TF2: {os.path.basename(path)}")
            base_model = tf.keras.applications.MobileNet(
                input_shape=(224,224,3), include_top=False, weights=None, pooling='avg'
            )
            x = tf.keras.layers.Dense(2, activation='softmax')(base_model.output)
            model_new = tf.keras.models.Model(inputs=base_model.input, outputs=x)
            try:
                model_new.load_weights(path, by_name=True, skip_mismatch=True)
                new_path = path.replace(".h5", "_tf2.h5")
                model_new.save(new_path)
                st.success(f"✅ Modelo convertido y guardado como {os.path.basename(new_path)}")
                return model_new
            except:
                st.error("❌ No se pudo cargar pesos, modelo reconstruido desde cero")
                return model_new
        else:
            raise e

# ---------- Modelos preentrenados ----------
folder_models = {}
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

for f in os.listdir(MODEL_DIR):
    if f.endswith(".h5"):
        folder_models[f] = os.path.join(MODEL_DIR, f)

select_options = [f"[Preentrenado] {n}" for n in folder_models.keys()] + \
                 [f"[Subido] {n}" for n in st.session_state.session_models.keys()]

selected_model_label = st.sidebar.selectbox("Selecciona un modelo", select_options)

# ---------- Cargar modelo seleccionado ----------
model = None
if selected_model_label:
    if selected_model_label.startswith("[Preentrenado]"):
        selected_model_name = selected_model_label.replace("[Preentrenado] ","")
        model_path = folder_models[selected_model_name]
    else:
        selected_model_name = selected_model_label.replace("[Subido] ","")
        model_path = st.session_state.session_models[selected_model_name]

    with st.spinner(f"Cargando modelo {selected_model_name}..."):
        model = load_model_safe(model_path)
    st.sidebar.success(f"✅ Modelo cargado")

st.write("---")

# ---------- Subir imágenes ----------
st.markdown("📂 Sube imágenes (JPG, PNG)")
uploaded_images = st.file_uploader("", type=['jpg','jpeg','png'], accept_multiple_files=True)

if uploaded_images:
    for img_file in uploaded_images:
        try:
            image_pil = Image.open(img_file)
            st.subheader(f"📷 {img_file.name}")
            st.image(image_pil, use_container_width=True)

            if model:
                img = image_pil.convert('RGB').resize(IMG_SIZE)
                img_array = np.expand_dims(np.array(img)/255.0, axis=0)
                preds = model.predict(img_array)

                if preds.ndim==2 and preds.shape[1]>1:
                    idx = np.argmax(preds[0])
                    conf = float(preds[0][idx])
                    label = CLASSES[idx]
                else:
                    val = float(preds.ravel()[0])
                    label = CLASSES[1] if val>=0.5 else CLASSES[0]
                    conf = val if val>=0.5 else 1-val

                st.success(f"✅ Predicción: **{label}** (confianza: {conf*100:.2f}%)")
            else:
                st.warning("⚠️ No hay modelo cargado")
        except Exception as e:
            st.error(f"❌ No se pudo procesar {img_file.name}: {e}")
else:
    st.info("ℹ️ Sube imágenes para ver predicciones")
