import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class AccuracyHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_accuracies = []
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_accuracies.append(logs.get('accuracy'))
        self.val_accuracies.append(logs.get('val_accuracy'))


# ====================
# CONFIGURAÇÕES INICIAIS
# ====================
csv_path = "roi_data.csv"
image_dir = "Rois/Normalizadas/"


# ====================
# 1. PREPROCESSAMENTO
# ====================
def load_and_preprocess_image(file_path, target_size, image_dir):
    file_path = os.path.join(image_dir, file_path)
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.keras.applications.mobilenet.preprocess_input(image)  # Normalização para MobileNet
    return image


# ====================
# 2. DEFINIR O MODELO COM FINE-TUNING
# ====================
def create_mobilenet_model(num_classes, learning_rate, convolucao, dropout):
    base_model = tf.keras.applications.MobileNet(weights="imagenet", include_top=False, input_shape=convolucao)
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ====================
# 3. VALIDAÇÃO CRUZADA
# ====================
def train_and_evaluate(input_size, learning_rate, convolucao, dropout, batch_size, epochs):
    data = pd.read_csv(csv_path)
    data["Paciente"] = data["Arquivo"].str.extract(r"ROI_(\d+)_\d+", expand=False).astype(int)
    pacientes = data["Paciente"].unique()

    test_accuracies = []
    train_accuracies = []
    conf_matrices = []
    class_weights = {0: 3.24, 1: 1.45}
    train_epoch_accuracies = []  # To store training accuracies per epoch for all folds
    test_epoch_accuracies = []  # To store testing accuracies per epoch for all folds
    model = create_mobilenet_model(
        num_classes=2, learning_rate=learning_rate, convolucao=convolucao, dropout=dropout
    )
    for fold, test_patient_id in enumerate(pacientes, 1):
        print(f"\nFold {fold}/{len(pacientes)}")
        test_data = data[data["Paciente"] == test_patient_id]
        train_data = data[data["Paciente"] != test_patient_id]

        x_train = np.array([load_and_preprocess_image(f, input_size, image_dir) for f in train_data["Arquivo"]])
        y_train = train_data["Classe"].values
        x_test = np.array([load_and_preprocess_image(f, input_size, image_dir) for f in test_data["Arquivo"]])
        y_test = test_data["Classe"].values

        accuracy_history = AccuracyHistory()
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            accuracy_history  # Custom callback to track epoch-wise accuracies
        ]

        model.fit(
            x_train, y_train, validation_data=(x_test, y_test),
            epochs=epochs, batch_size=batch_size, verbose=0,
            class_weight=class_weights, callbacks=callbacks
        )

        train_epoch_accuracies.append(accuracy_history.train_accuracies)
        test_epoch_accuracies.append(accuracy_history.val_accuracies)

        preds = model.predict(x_test, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        test_accuracies.append(accuracy_score(y_test, y_pred))

        conf_matrices.append(confusion_matrix(y_test, y_pred))

    return test_accuracies, train_epoch_accuracies, test_epoch_accuracies, conf_matrices, model


import random


def generate_random_seeds(size=9, seed_range=(0, 2 ** 32 - 1)):
    return [random.randint(seed_range[0], seed_range[1]) for _ in range(size)]


# ====================
# 4. GRID SEARCH PARA HIPERPARÂMETROS
# ====================
def save_best_params_to_file(file_path, params, test_accuracies, train_accuracies, mean_accuracy_test):
    # Formatar os parâmetros e a acurácia para salvar no arquivo
    result_line = f"mean accuracy test:{mean_accuracy_test}\ntest Accuracy: {test_accuracies}\ntrain Accuracy: {train_accuracies}\n Params: {params}\n"
    with open(file_path, "w") as file:
        file.write(result_line)


def grid_search():
    learning_rates = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    batch_sizes = [55]
    epochs_list = [40, 60]
    input_sizes_and_convolucao = {(128, 128): (128, 128, 3)}
    dropout_rates = [0.3, 0.5, 0.7, 0]
    best_params = {}
    seed_vector = generate_random_seeds()
    path_result = "melhores.txt"
    best_accuracy = 0
    for lr, batch_size, epochs, (input_size, convolucao), dropout, seed in itertools.product(
            learning_rates, batch_sizes, epochs_list, input_sizes_and_convolucao.items(), dropout_rates, seed_vector
    ):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(
            f"Testando: LR={lr}, Batch size={batch_size}, Epochs={epochs}, Input size={input_size}, Dropout={dropout}, seed={seed}")
        start_time = time.time()

        test_accuracies, train_accuracy_epoch, test_accuracy_epoch, conf_matrices, model = train_and_evaluate(
            input_size, lr, convolucao, dropout, batch_size, epochs
        )
        mean_accuracy = np.mean(test_accuracies)
        elapsed_time = time.time() - start_time
        print(f"Resultado: Acurácia Média={mean_accuracy:.4f}, Tempo={elapsed_time:.2f}s")
        params = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "input_size": input_size,
            "dropout": dropout,
            "seed": seed,
        }
        if mean_accuracy > best_accuracy:
            salvaGeral(mean_accuracy, params, model, test_accuracy_epoch, train_accuracy_epoch, conf_matrices,
                       path_result)
            best_accuracy = mean_accuracy
            best_params = params

    print("\nMelhor Configuração:")
    print(f"Acurácia: {best_accuracy:.4f}")
    print(f"Hiperparâmetros: {best_params}")


def salvaGeral(mean_accuracy, params, model, test_accuracies, train_accuracy, conf_matrices, path_result):
    save_best_params_to_file(path_result, params, test_accuracies, train_accuracy, mean_accuracy)
    save_confusion_matrices_to_txt(conf_matrices)
    model.save("MobileNet.h5")


def save_confusion_matrices_to_txt(conf_matrices, output_file="confusion_matrices.txt"):
    with open(output_file, "w") as file:
        for i, cm in enumerate(conf_matrices, start=1):
            file.write(f"Fold {i}:\n")
            np.savetxt(file, cm, fmt="%d", delimiter=",")
            file.write("\n")  # Separate matrices with a blank line
    print(f"Confusion matrices saved to {output_file}")


# Executar o Grid Search
grid_search()
