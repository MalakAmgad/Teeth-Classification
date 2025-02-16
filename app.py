import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


# Class labels (ensure this matches the order used in training)
class_labels = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Load the trained model
@st.cache_resource
def load_model():
    st.write("Loading model...")  # Debugging line
    model = tf.keras.models.load_model('fine_tuned_cnn_teeth_classifier.h5') 
    st.write("Model loaded!")  # Debugging line
    return model

model = load_model()

# Image Preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = img_to_array(img)  # Now correctly using Keras function
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Streamlit App
st.title("Teeth Classification")

uploaded_file = st.file_uploader("Upload an Image of a Tooth", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.write("Image uploaded!")  # Debugging line
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)  
    
    st.write("Classifying...")
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
