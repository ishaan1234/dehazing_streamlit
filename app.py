import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Function to load and preprocess the image
def load_image(image):
    img = Image.open(image).convert('RGB')
    st.image(img)
    st.write("Generating Pix2Pix Image...")
    with st.spinner("Converting..."):
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = (img / 127.5) - 1.0
        img = tf.image.resize(img, [256, 256])
        input_image = tf.expand_dims(img, 0)
        model = tf.keras.models.load_model(r"twoghaze1k.h5")
        prediction = model(input_image, training=True)
        res = prediction[0]* 0.5 + 0.5
    st.image(res.numpy())

# Streamlit UI
def main():
    st.title("Satellite Image Dehazing")
    st.write("Upload a hazy image and let the model do the magic!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.write("Original Image:")
        load_image(uploaded_file)
        

        
        # model = load_model()
        # pix2pix_image = generate_image(model, original_image)
        # pix2pix_image = (pix2pix_image + 1) / 2  # Convert back to [0, 1] range 
        # st.write("Pix2Pix Converted Image:")
        # st.image(pix2pix_image.numpy(), use_column_width=True)

if __name__ == "__main__":
    main()
