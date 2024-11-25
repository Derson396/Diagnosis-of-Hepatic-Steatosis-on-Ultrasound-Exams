import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from PIL import Image

# ====================
# LOAD THE MODEL
# ====================
model_path = "MobileNet.h5"  # Path to the saved model
model = load_model(model_path)


# ====================
# IMAGE PREPROCESSING
# ====================
def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess a single image for prediction.

    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Target size for resizing the image.

    Returns:
        numpy.ndarray: Preprocessed image ready for model prediction.
    """
    image = Image.open(image_path).convert("RGB")  # Ensure it's an RGB image
    image = image.resize(target_size)  # Resize the image
    image = np.array(image)  # Convert to a NumPy array
    image = preprocess_input(image)  # Normalize the image using MobileNet preprocessing
    return np.expand_dims(image, axis=0)  # Add batch dimension


# ====================
# PREDICT CLASS
# ====================
def predict_class(image_path):
    """
    Predict the class of an image using the MobileNet model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        int: Predicted class label.
        numpy.ndarray: Probabilities for each class.
    """
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class, predictions


# ====================
# MAIN EXECUTION
# ====================
if __name__ == "__main__":
    # Specify the image path
    input_image_path = "Rois/Normalizadas/ROI_0_1.png"  # Update with your image path

    # Check if the file exists
    if not os.path.exists(input_image_path):
        print(f"Error: File '{input_image_path}' not found.")
    else:
        # Predict the class
        predicted_class, probabilities = predict_class(input_image_path)

        print(f"Predicted Class: {predicted_class}")
        print(f"Class Probabilities: {probabilities}")
