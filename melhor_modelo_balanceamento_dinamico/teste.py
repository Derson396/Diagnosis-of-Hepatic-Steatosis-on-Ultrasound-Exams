import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import random

# ====================
# LOAD THE MODEL
# ====================
model_path = "MobileNet_0.001_55.h5"
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
    image = load_and_preprocess_image(file_path, target_size)
    # Rotate image by multiples of 90 degrees
    image = tf.image.rot90(image, k=random_angle // 90)
    return image


def load_and_preprocess_image_with_rotation2(file_path, target_size, angle):
    """
    Apply rotation by 90°, 180°, or 270° to the image.
    """
    image = load_and_preprocess_image(file_path, target_size)

    if angle == 90:
        image = tf.image.rot90(image, k=1)
    elif angle == 180:
        image = tf.image.rot90(image, k=2)
    elif angle == 270:
        image = tf.image.rot90(image, k=3)

    return image


def load_and_preprocess_image_with_translation(file_path, target_size, max_translation=0.2):
    image = load_and_preprocess_image(file_path, target_size)

    # Calculando os limites de translação
    translation_x = tf.random.uniform([], -max_translation, max_translation) * tf.cast(target_size[1], tf.float32)
    translation_y = tf.random.uniform([], -max_translation, max_translation) * tf.cast(target_size[0], tf.float32)

    # Aplica a translação usando a função affine_transform
    image = tf.image.resize_with_crop_or_pad(image, target_size[0] + int(translation_y),
                                             target_size[1] + int(translation_x))

    # Aplica o recorte ou preenchimento para voltar ao tamanho alvo, garantindo que a imagem seja do tamanho correto
    image = tf.image.resize(image, target_size)

    return image


def load_and_preprocess_image_with_scaling(file_path, target_size, min_scale=0.8, max_scale=1.2):
    image = load_and_preprocess_image(file_path, target_size)

    # Escolher um fator de escala aleatório
    scale_factor = tf.random.uniform([], min_scale, max_scale)

    # Redimensiona a imagem com base no fator de escala
    new_size = tf.cast(tf.round(tf.cast(target_size, tf.float32) * scale_factor), tf.int32)
    image = tf.image.resize(image, new_size)

    # Redimensiona novamente para o tamanho alvo final
    image = tf.image.resize(image, target_size)

    return image


def load_and_preprocess_image_with_flip(file_path, target_size):
    image = load_and_preprocess_image(file_path, target_size)

    # Flip horizontal ou vertical aleatório
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image


def load_and_preprocess_image_with_brightness_contrast(file_path, target_size, max_delta=0.2):
    image = load_and_preprocess_image(file_path, target_size)

    # Ajusta brilho e contraste aleatórios
    image = tf.image.random_brightness(image, max_delta)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    return image


# ====================
# EVALUATION FUNCTIONS
# ====================
def evaluate_images(image_dir, target_size, patient_split, rotate=False, rotate2=False, translate=False, scale=False, flip=False, brightness_contrast=False):
    """
    Evaluate model predictions on images, optionally with random transformations.

    Args:
        image_dir (str): Directory containing images.
        target_size (tuple): Target size for resizing (height, width).
        patient_split (dict): Dictionary mapping healthy and diseased patients.
        rotate (bool): Whether to apply random rotations to the images.
        rotate2 (bool): Whether to apply specific 90°, 180°, 270° rotations.
        translate (bool): Whether to apply random translations to the images.
        scale (bool): Whether to apply random scaling to the images.
        flip (bool): Whether to apply random flipping to the images.
        brightness_contrast (bool): Whether to apply random brightness and contrast adjustments.

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
        elif rotate2:
            angle = random.choice([90, 180, 270])
            preprocessed_image = load_and_preprocess_image_with_rotation2(image_path, target_size, angle)
        elif translate:
            preprocessed_image = load_and_preprocess_image_with_translation(image_path, target_size)
        elif scale:
            preprocessed_image = load_and_preprocess_image_with_scaling(image_path, target_size)
        elif flip:
            preprocessed_image = load_and_preprocess_image_with_flip(image_path, target_size)
        elif brightness_contrast:
            preprocessed_image = load_and_preprocess_image_with_brightness_contrast(image_path, target_size)
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
            misclassified_images.append((
                file_name,
                predicted_class,
                ground_truth,
                "Rotated" if rotate else (
                    "Rotated2" if rotate2 else (
                        "Translated" if translate else (
                            "Scaled" if scale else (
                                "Flipped" if flip else (
                                    "Brightness/Contrast" if brightness_contrast else "Normal"
                                )
                            )
                        )
                    )
                )
            ))

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
    normal_accuracy, normal_misclassified = evaluate_images(image_dir, target_size, patient_split, rotate=False, rotate2=False, translate=False, scale=False, flip=False, brightness_contrast=False)
    print(f"Accuracy on normal images: {normal_accuracy:.2f}%")

    # Evaluate images with random rotations
    rotated_accuracy, rotated_misclassified = evaluate_images(image_dir, target_size, patient_split, rotate=True, rotate2=False, translate=False, scale=False, flip=False, brightness_contrast=False)
    print(f"Accuracy on randomly rotated images: {rotated_accuracy:.2f}%")

    # Evaluate images with specific rotations (90°, 180°, 270°)
    rotated2_accuracy, rotated2_misclassified = evaluate_images(image_dir, target_size, patient_split, rotate=False, rotate2=True, translate=False, scale=False, flip=False, brightness_contrast=False)
    print(f"Accuracy on images with specific rotations (90°, 180°, 270°): {rotated2_accuracy:.2f}%")

    # Evaluate images with random translations
    translated_accuracy, translated_misclassified = evaluate_images(image_dir, target_size, patient_split, rotate=False, rotate2=False, translate=True, scale=False, flip=False, brightness_contrast=False)
    print(f"Accuracy on randomly translated images: {translated_accuracy:.2f}%")

    # Evaluate images with random scaling
    scaled_accuracy, scaled_misclassified = evaluate_images(image_dir, target_size, patient_split, rotate=False, rotate2=False, translate=False, scale=True, flip=False, brightness_contrast=False)
    print(f"Accuracy on randomly scaled images: {scaled_accuracy:.2f}%")

    # Evaluate images with random flips
    flipped_accuracy, flipped_misclassified = evaluate_images(image_dir, target_size, patient_split, rotate=False, rotate2=False, translate=False, scale=False, flip=True, brightness_contrast=False)
    print(f"Accuracy on randomly flipped images: {flipped_accuracy:.2f}%")

    # Evaluate images with random brightness and contrast
    brightness_contrast_accuracy, brightness_contrast_misclassified = evaluate_images(image_dir, target_size, patient_split, rotate=False, rotate2=False, translate=False, scale=False, flip=False, brightness_contrast=True)
    print(f"Accuracy on images with random brightness/contrast: {brightness_contrast_accuracy:.2f}%")

    # Print misclassified images
    if normal_misclassified:
        print("\nMisclassified normal images:")
        for file_name, predicted_class, ground_truth, _ in normal_misclassified:
            print(f"- {file_name}: Predicted={predicted_class}, True={ground_truth}")

    if rotated_misclassified:
        print("\nMisclassified rotated images:")
        for file_name, predicted_class, ground_truth, mode in rotated_misclassified:
            print(f"- {file_name}: Predicted={predicted_class}, True={ground_truth}, Mode={mode}")

    if rotated2_misclassified:
        print("\nMisclassified rotated2 images:")
        for file_name, predicted_class, ground_truth, mode in rotated2_misclassified:
            print(f"- {file_name}: Predicted={predicted_class}, True={ground_truth}, Mode={mode}")

    if translated_misclassified:
        print("\nMisclassified translated images:")
        for file_name, predicted_class, ground_truth, mode in translated_misclassified:
            print(f"- {file_name}: Predicted={predicted_class}, True={ground_truth}, Mode={mode}")

    if scaled_misclassified:
        print("\nMisclassified scaled images:")
        for file_name, predicted_class, ground_truth, mode in scaled_misclassified:
            print(f"- {file_name}: Predicted={predicted_class}, True={ground_truth}, Mode={mode}")

    if flipped_misclassified:
        print("\nMisclassified flipped images:")
        for file_name, predicted_class, ground_truth, mode in flipped_misclassified:
            print(f"- {file_name}: Predicted={predicted_class}, True={ground_truth}, Mode={mode}")

    if brightness_contrast_misclassified:
        print("\nMisclassified brightness/contrast images:")
        for file_name, predicted_class, ground_truth, mode in brightness_contrast_misclassified:
            print(f"- {file_name}: Predicted={predicted_class}, True={ground_truth}, Mode={mode}")

    # Comparison
    print("\nComparison:")
    print(f"Normal Accuracy: {normal_accuracy:.2f}%")
    print(f"Rotated Accuracy: {rotated_accuracy:.2f}%")
    print(f"Rotated2 Accuracy: {rotated2_accuracy:.2f}%")
    print(f"Translated Accuracy: {translated_accuracy:.2f}%")
    print(f"Scaled Accuracy: {scaled_accuracy:.2f}%")
    print(f"Flipped Accuracy: {flipped_accuracy:.2f}%")
    print(f"Brightness/Contrast Accuracy: {brightness_contrast_accuracy:.2f}%")
