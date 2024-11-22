import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools
import time
import matplotlib.pyplot as plt
import seaborn as sns

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
    image = preprocess_input(image)  # Normalização para MobileNet
    return image


# ====================
# 2. DEFINIR O MODELO COM FINE-TUNING
# ====================
def create_mobilenet_model(num_classes, learning_rate, convolucao, dropout):
    base_model = MobileNet(weights="imagenet", include_top=False, input_shape=convolucao)
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    output = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=output)
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

    accuracies = []
    conf_matrices = []

    class_weights = {0: 3.24, 1: 1.45}
    for fold, test_patient_id in enumerate(pacientes, 1):
        print(f"\nFold {fold}/{len(pacientes)}")
        test_data = data[data["Paciente"] == test_patient_id]
        train_data = data[data["Paciente"] != test_patient_id]

        x_train = np.array([load_and_preprocess_image(f, input_size, image_dir) for f in train_data["Arquivo"]])
        y_train = train_data["Classe"].values
        x_test = np.array([load_and_preprocess_image(f, input_size, image_dir) for f in test_data["Arquivo"]])
        y_test = test_data["Classe"].values

        model = create_mobilenet_model(
            num_classes=2, learning_rate=learning_rate, convolucao=convolucao, dropout=dropout
        )

        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        ]

        model.fit(
            x_train, y_train, validation_data=(x_test, y_test),
            epochs=epochs, batch_size=batch_size, verbose=0,
            class_weight=class_weights, callbacks=callbacks
        )

        preds = model.predict(x_test, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        accuracies.append(accuracy_score(y_test, y_pred))
        conf_matrices.append(confusion_matrix(y_test, y_pred))

    return np.mean(accuracies), np.std(accuracies), conf_matrices
import random
def generate_random_seeds(size=9, seed_range=(0, 2**32 - 1)):
    return [random.randint(seed_range[0], seed_range[1]) for _ in range(size)]
# ====================
# 4. GRID SEARCH PARA HIPERPARÂMETROS
# ====================
def save_best_params_to_file(file_path, params, accuracy):
    # Formatar os parâmetros e a acurácia para salvar no arquivo
    result_line = f"Accuracy: {accuracy:.4f}, Params: {params}\n"
    with open(file_path, "a") as file:
        file.write(result_line)

def grid_search():
    learning_rates = [1e-3, 1e-4, 1e-5,1e-6,1e-7,1e-8]
    batch_sizes = [55]
    epochs_list = [40, 60]
    input_sizes_and_convolucao = {(32, 32): (32, 32, 3)}
    dropout_rates = [0.3, 0.5, 0.7,0]
    best_accuracy = 0
    best_params = {}
    seed_vector = generate_random_seeds()
    path_result = "melhores.txt"
    for lr, batch_size, epochs, (input_size, convolucao), dropout,seed in itertools.product(
        learning_rates, batch_sizes, epochs_list, input_sizes_and_convolucao.items(), dropout_rates,seed_vector
    ):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(f"Testando: LR={lr}, Batch size={batch_size}, Epochs={epochs}, Input size={input_size}, Dropout={dropout}, seed={seed}")
        start_time = time.time()

        mean_accuracy, std_accuracy, conf_matrices = train_and_evaluate(
            input_size, lr, convolucao, dropout, batch_size, epochs
        )

        elapsed_time = time.time() - start_time
        print(f"Resultado: Acurácia Média={mean_accuracy:.4f}, Desvio Padrão={std_accuracy:.4f}, Tempo={elapsed_time:.2f}s")
        params = {
            "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "input_size": input_size,
                "dropout": dropout,
                "seed":seed,
        }
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = params
        if mean_accuracy>0.88:
            save_best_params_to_file(path_result,params,mean_accuracy)

    print("\nMelhor Configuração:")
    print(f"Acurácia: {best_accuracy:.4f}")
    print(f"Hiperparâmetros: {best_params}")


# Executar o Grid Search
grid_search()
