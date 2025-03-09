import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load multi-class model
multi_class_model = tf.keras.models.load_model("cnn_model.h5")

# Class labels with full names
disease_mapping = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevi",
    "BCC": "Basal Cell Carcinoma",
    "AKIEC": "Actinic Keratoses and Intraepithelial Carcinoma (Bowen's Disease)",
    "BKL": "Benign Keratosis-like Lesions",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesions"
}

# Load all binary classification models
binary_models = {disease: tf.keras.models.load_model(f"multiple_binary_classifier/{disease}_model.h5") for disease in disease_mapping.keys()}

def predict_image(uploaded_file):
    # Convert uploaded file to a NumPy array
    image = Image.open(uploaded_file).convert("RGB")  # Open image in RGB mode
    image = image.resize((150, 200))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Expand dims for model

    # Use Multi-Class Model for Initial Prediction
    predictions = multi_class_model.predict(image)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    
    # Get the predicted disease full name
    predicted_disease = list(disease_mapping.keys())[class_index]
    predicted_disease_full_name = disease_mapping[predicted_disease]

    # If confidence is low (< 0.6), use binary model for verification
    if confidence < 0.6:
        binary_model = binary_models[predicted_disease]
        binary_pred = binary_model.predict(image)
        binary_confidence = binary_pred[0][0]  # Binary models output probability
        
        # Adjust confidence based on binary model
        if binary_confidence >= 0.5:
            confidence = binary_confidence  # Use binary model's confidence
        else:
            predicted_disease_full_name = "Uncertain (Re-check Needed)"
            confidence = binary_confidence

    return predicted_disease_full_name, confidence
