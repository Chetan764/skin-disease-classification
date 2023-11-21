import streamlit as st
import tensorflow as tf
from tf.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model(r'best-model1.h5')

# Define the classes
classes = ['Earmites', 'Flea_allergy', 'Healthy', 'Leprosy', 'Pyoderma', 'Ringworm']

# Create a Streamlit web app
st.title("Pet Scan")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)

    # Display the top prediction
    class_index = np.argmax(predictions)
    class_prediction = classes[class_index]

    st.write("Prediction:")
    st.write(f"This image is classified as {class_prediction} with confidence: {predictions[0][class_index]:.2f}")
