import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ====================
# LOAD THE MODEL
# ====================
model_path = "MobileNet.h5"  # Path to the saved model
model = load_model(model_path)

# ====================
# IMAGE PREPROCESSING
# ====================
def load_and_preprocess_image(file_path, target_size=(128, 128), image_dir=""):
    """
    Preprocess a single image for prediction.

    Args:
        file_path (str): Name of the input image file.
        target_size (tuple): Target size for resizing the image.
        image_dir (str): Directory where the image file is located.

    Returns:
        numpy.ndarray: Preprocessed image ready for model prediction.
    """
    file_path = os.path.join(image_dir, file_path)
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)  # Ensure 3 color channels (RGB)
    image = tf.image.resize(image, target_size)  # Resize the image
    image = tf.keras.applications.mobilenet.preprocess_input(image)  # MobileNet preprocessing
    return np.expand_dims(image.numpy(), axis=0)  # Add batch dimension

# ====================
# AUTOMATED TESTING
# ====================
def evaluate_model_on_dataset(normalized_dir):
    """
    Evaluate the model on all images in the directory and calculate accuracy.

    Args:
        normalized_dir (str): Directory containing the normalized images.

    Returns:
        None
    """
    correct_predictions = 0
    total_images = 0
    errors = []

    for file_name in os.listdir(normalized_dir):
        # Check if the file is a valid image
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Extract patient ID from the file name
        patient_id = int(file_name.split('_')[1])
        expected_class = 0 if patient_id <= 16 else 1  # Healthy: 0, Diseased: 1

        # Predict the class
        processed_image = load_and_preprocess_image(file_name, target_size=(128, 128), image_dir=normalized_dir)
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Check if the prediction is correct
        total_images += 1
        if predicted_class == expected_class:
            correct_predictions += 1
        else:
            errors.append((file_name, expected_class, predicted_class, predictions))

    # Calculate accuracy
    accuracy = correct_predictions / total_images * 100
    print(f"\nTotal Images: {total_images}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Print errors
    if errors:
        print("\nErrors:")
        for file_name, expected, predicted, probs in errors:
            print(f"File: {file_name}, Expected: {expected}, Predicted: {predicted}, Probabilities: {probs}")
            # Optional: Display the misclassified images
            image_path = os.path.join(normalized_dir, file_name)
            image = tf.io.read_file(image_path)
            image = tf.image.decode_png(image, channels=3).numpy()
            #plt.imshow(image.astype(np.uint8))
            #plt.title(f"File: {file_name}\nExpected: {expected}, Predicted: {predicted}")
            #plt.axis("off")
            #plt.show()

# ====================
# MAIN EXECUTION
# ====================
if __name__ == "__main__":
    # Specify the directory containing normalized images
    normalized_dir = "Rois/Normalizadas"  # Update with the path to your images

    # Check if the directory exists
    if not os.path.exists(normalized_dir):
        print(f"Error: Directory '{normalized_dir}' not found.")
    else:
        # Evaluate the model on the dataset
        evaluate_model_on_dataset(normalized_dir)
