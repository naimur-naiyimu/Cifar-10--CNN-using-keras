import streamlit as st
import numpy as np
from PIL import Image
import os
import pickle

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

st.set_page_config(page_title="CIFAR-10 Image Classification", page_icon="🖼️", layout="centered")

# Define class names for CIFAR-10
CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

@st.cache_resource
def load_model():
    model_path = 'cifar10_model_88.keras'
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    # Convert image to RGB if it has an alpha channel
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize to 32x32 as required by CIFAR-10 models
    image = image.resize((32, 32))
    
    if HAS_TF:
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    else:
        # Fallback dummy array
        return np.random.rand(1, 32, 32, 3)

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("About")
        st.info(
            "This Convolutional Neural Network (CNN) model was trained on the CIFAR-10 dataset to classify "
            "images into 10 distinct categories with high accuracy."
        )
        st.markdown("---")
        st.markdown("### Developed By")
        st.markdown("**Naimur Rahman**")
        st.markdown("---")

    st.title("CIFAR-10 Image Classifier")
    st.markdown("""
    Welcome to the CIFAR-10 Image Classifier! 
    This model was trained to recognize 10 different categories:
    **Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck**.
    """)

    # --- Training History Section ---
    pkl_path = 'cifar10_88.pkl'
    if os.path.exists(pkl_path):
        with st.expander("📈 View Training History", expanded=False):
            st.info("The  model's training history (accuracy and loss)")
            try:
                with open(pkl_path, 'rb') as f:
                    history = pickle.load(f)
                
                # Plot accuracy
                if 'accuracy' in history and 'val_accuracy' in history:
                    st.write("**Model Accuracy**")
                    acc_data = {"Train Accuracy": history['accuracy'], "Validation Accuracy": history['val_accuracy']}
                    st.line_chart(acc_data)
                
                # Plot loss
                if 'loss' in history and 'val_loss' in history:
                    st.write("**Model Loss**")
                    loss_data = {"Train Loss": history['loss'], "Validation Loss": history['val_loss']}
                    st.line_chart(loss_data)
            except Exception as e:
                st.error(f"Could not load training history from {pkl_path}: {e}")

    st.markdown("---")
    
    if not HAS_TF:
        st.warning("⚠️ **TensorFlow is required to load the `.keras` model.** Please wait for the background installation to finish or run `pip install tensorflow`.")
        st.stop()
        
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Model file `cifar10_model_88.keras` not found or failed to load.")
        st.stop()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        st.write("Classifying...")
        
        # Preprocess the image (resizes to 32x32)
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)

        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        # Display results
        st.success(f"### Prediction: {CLASS_NAMES[predicted_class_index]}")
        st.info(f"Confidence: {confidence * 100:.2f}%")
        
        # Show probability distribution
        st.write("#### Prediction Probabilities:")
        
        # Create a progress bar style visualization for probabilities
        cols = st.columns(2)
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions[0])):
            with cols[i % 2]:
                st.write(f"{class_name}: {prob*100:.2f}%")
                st.progress(float(prob))

if __name__ == '__main__':
    main()
