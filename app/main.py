import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class names
class_indices = json.load(open(os.path.join(working_dir, "class_indices.json")))

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Center the title with a Bootstrap leaf icon
st.markdown(
    """
    <style>
    .centered-title {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .centered-title img {
        margin-right: 10px;
    }
    </style>
    <div class="centered-title">
        <h1>Plant Disease Classifier</h1>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns([1, 2])

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        st.write("")  # Add some space between the image and the button
        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
