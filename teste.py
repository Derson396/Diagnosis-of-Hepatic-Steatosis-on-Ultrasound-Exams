import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import random

# ====================
# LOAD THE MODEL
# ====================
model_path = "MobileNet.h5"
model = load_model(model_path)


# ====================
# IMAGE PREPROCESSING
# ====================
def load_and_preprocess_image(file_path, target_size):
    """
    Load and preprocess an image for prediction.

    Args:
        file_path (str): Path to the image.
        target_size (tuple): Target size for resizing (height, width).

    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.keras.applications.mobilenet.preprocess_input(image)
    return image


def load_and_preprocess_image_with_rotation(file_path, target_size, random_angle):
    """
    Load and preprocess an image with random rotation.

    Args:
        file_path (str): Path to the image.
        target_size (tuple): Target size for resizing (height, width).
        random_angle (float): Random rotation angle in degrees.

    Returns:
        tf.Tensor: Preprocessed and rotated image tensor.
    """
    image = load_and_preprocess_image(file_path, target_size)
    # Rotate image by multiples of 90 degrees
    image = tf.image.rot90(image, k=random_angle // 90)
    return image


# ====================
# EVALUATION FUNCTIONS
# ====================
def evaluate_images(image_dir, target_size, patient_split, rotate=False):
    """
    Evaluate model predictions on images, optionally with random rotations.

    Args:
        image_dir (str): Directory containing images.
        target_size (tuple): Target size for resizing (height, width).
        patient_split (dict): Dictionary mapping healthy and diseased patients.
        rotate (bool): Whether to apply random rotations to the images.

    Returns:
        float: Accuracy of the model on the evaluated images.
        list: Misclassified images.
    """
    healthy_patients = patient_split["healthy"]
    diseased_patients = patient_split["diseased"]

    total_images = 0
    correct_predictions = 0
    misclassified_images = []

    for file_name in os.listdir(image_dir):
        # Determine the ground truth label
        patient_id = int(file_name.split("_")[1])  # Extract patient number from filename
        ground_truth = 0 if patient_id in healthy_patients else 1

        # Preprocess the image
        image_path = os.path.join(image_dir, file_name)
        if rotate:
            random_angle = random.choice([0, 90, 180, 270])
            preprocessed_image = load_and_preprocess_image_with_rotation(image_path, target_size, random_angle)
        else:
            preprocessed_image = load_and_preprocess_image(image_path, target_size)
        preprocessed_image = tf.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

        # Predict the class
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Check if the prediction matches the ground truth
        if predicted_class == ground_truth:
            correct_predictions += 1
        else:
            misclassified_images.append((file_name, predicted_class, ground_truth, "Rotated" if rotate else "Normal"))

        total_images += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_images * 100
    return accuracy, misclassified_images


# ====================
# MAIN EXECUTION
# ====================
if __name__ == "__main__":
    image_dir = "Rois/Normalizadas"
    target_size = (128, 128)
    patient_split = {
        "healthy": range(0, 17),  # Patients 0-16 are healthy
        "diseased": range(17, 100)  # Patients 17+ are diseased
    }

    # Evaluate normal images
    normal_accuracy, normal_misclassified = evaluate_images(image_dir, target_size, patient_split, rotate=False)
    print(f"Accuracy on normal images: {normal_accuracy:.2f}%")

    # Evaluate images with random rotations
    rotated_accuracy, rotated_misclassified = evaluate_images(image_dir, target_size, patient_split, rotate=True)
    print(f"Accuracy on randomly rotated images: {rotated_accuracy:.2f}%")


    # Print misclassified images
    if normal_misclassified:
        print("\nMisclassified normal images:")
        for file_name, predicted_class, ground_truth, _ in normal_misclassified:
            print(f"- {file_name}: Predicted={predicted_class}, True={ground_truth}")

    if rotated_misclassified:
        print("\nMisclassified rotated images:")
        for file_name, predicted_class, ground_truth, mode in rotated_misclassified:
            print(f"- {file_name}: Predicted={predicted_class}, True={ground_truth}, Mode={mode}")

    # Compare results
    print("\nComparison:")
    print(f"Normal Accuracy: {normal_accuracy:.2f}%")
    print(f"Rotated Accuracy: {rotated_accuracy:.2f}%")
    print(f"Difference: {normal_accuracy - rotated_accuracy:.2f}%")
