import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Available models
MODELS = {
    "Apple": "apple",
    "Cherry": "cherry",
    "Corn": "corn",
    "Cassava": "cassava",
    "Bell Pepper": "bell_pepper",
    "Grape": "grape"
}

@st.cache_resource
def load_model(model_name):
    model_path = f"hydra_models/{model_name}/model.tflite"
    labels_path = f"hydra_models/{model_name}/labels.txt"

    if not os.path.exists(model_path):
        st.error(f"❌ Model file missing: {model_path}")
        return None, None

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    with open(labels_path, 'r') as f:
        labels = [line.strip().split(maxsplit=1)[-1] for line in f.readlines()]

    return interpreter, labels

def classify_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    image = image.resize((input_shape[2], input_shape[1]))
    input_data = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    top_index = np.argmax(output_data)
    confidence = float(output_data[top_index])

    return top_index, confidence

# UI
st.set_page_config(page_title="Hydra AI", layout="centered")
st.title("🌿 Hydra AI - Plant Disease Classifier")

selected_model = st.selectbox("Choose a plant model", list(MODELS.keys()))
interpreter, labels = load_model(MODELS[selected_model])

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        index, confidence = classify_image(interpreter, image)
        label = labels[index] if labels else "Unknown"

    st.success(f"🩺 Prediction: **{label}** ({confidence*100:.2f}% confidence)")
