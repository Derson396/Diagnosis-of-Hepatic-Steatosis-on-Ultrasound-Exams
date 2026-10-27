import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
from actions.cross_validation import split_patient_validation

import warnings
warnings.filterwarnings("ignore")  # Ignora todos os warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Níveis: 0 (tudo), 1 (info), 2 (warnings), 3 (erros)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Mostra apenas mensagens de erro


class AccuracyHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_accuracies = []
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_accuracies.append(logs.get("accuracy"))
        self.val_accuracies.append(logs.get("val_accuracy"))

def calculate_sensitivity_specificity(conf_matrix):
    if conf_matrix.shape == (2, 2):
        TN, FP, FN, TP = conf_matrix.ravel()
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    else:
        sensitivity, specificity = 0, 0

    return sensitivity, specificity




def is_better_model(sensitivity, specificity, accuracy, best_model_score, alpha=0.4, beta=0.1, gamma=0.5):
    current_score = alpha * sensitivity + beta * specificity + gamma * accuracy
    if current_score > best_model_score:
        return True
    return False


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
    image = tf.keras.applications.mobilenet.preprocess_input(
        image
    )  # Normalização para MobileNet
    return image


# ====================
# 2. DEFINIR O MODELO COM FINE-TUNING
# ====================
def create_mobilenet_model(num_classes, learning_rate, convolucao, dropout):
    base_model = tf.keras.applications.MobileNet(
        weights="imagenet", include_top=False, input_shape=convolucao
    )
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
        metrics=["accuracy"],
    )
    return model


# ====================
# 3. VALIDAÇÃO CRUZADA
# ====================
def train_and_evaluate(
    input_size, learning_rate, convolucao, dropout, batch_size, epochs
):
    data = pd.read_csv(csv_path)
    data["Paciente"] = (
        data["Arquivo"].str.extract(r"ROI_(\d+)_\d+", expand=False).astype(int)
    )
    pacientes = data["Paciente"].unique()

    # Variáveis para acumular as predições e valores reais
    all_preds = []
    all_labels = []

    # Outras variáveis para armazenar métricas de cada fold (como sensibilidade e especificidade)
    test_accuracies = []
    sensibilities = []
    specificities = []
    conf_matrices = []

    # Inicialize as listas para armazenar as métricas por época
    train_accuracies = []
    test_accuracies = []

    #class_weights = {0: 3.24, 1: 1.45}


    train_epoch_accuracies = []
    test_epoch_accuracies = []
    for fold, test_patient_id in enumerate(pacientes, 1):
        print(f"\nFold {fold}/{len(pacientes)}")  # Pode ser removido para evitar prints
        test_data = data[data["Paciente"] == test_patient_id]
        train_data = data[data["Paciente"] != test_patient_id]
        fit_patients, validation_patients = split_patient_validation(train_data["Paciente"])
        validation_data = train_data[train_data["Paciente"].isin(validation_patients)]
        train_data = train_data[train_data["Paciente"].isin(fit_patients)]

        x_train = np.array(
            [
                load_and_preprocess_image(f, input_size, image_dir)
                for f in train_data["Arquivo"]
            ]
        )
        y_train = train_data["Classe"].values
        x_validation = np.array(
            [load_and_preprocess_image(f, input_size, image_dir) for f in validation_data["Arquivo"]]
        )
        y_validation = validation_data["Classe"].values
        x_test = np.array(
            [
                load_and_preprocess_image(f, input_size, image_dir)
                for f in test_data["Arquivo"]
            ]
        )
        y_test = test_data["Classe"].values

        model = create_mobilenet_model(
            num_classes=2,
            learning_rate=learning_rate,
            convolucao=convolucao,
            dropout=dropout,
        )
        accuracy_history = AccuracyHistory()
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            accuracy_history,
        ]

        # Cálculo dinâmico do class_weight
        classes = np.unique(y_train)
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weights = dict(zip(classes, class_weights))

        model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,  # Evita mensagens de treinamento
            class_weight=class_weights,  # Usa pesos dinâmicos
            callbacks=callbacks,
        )

        # Prevendo para os dados de teste
        preds = model.predict(x_test, verbose=0)
        y_pred = np.argmax(preds, axis=1)

        # Acumulando as predições e valores reais para todos os folds
        all_preds.extend(y_pred)
        all_labels.extend(y_test)

        # Calculando métricas para cada fold (não necessariamente necessário para métricas totais)
        # cm = confusion_matrix(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape != (2, 2):
            cm_resized = np.zeros((2, 2), dtype=int)
            # Preencher a matriz 2x2 com as informações disponíveis
            for i in range(min(2, cm.shape[0])):
                for j in range(min(2, cm.shape[1])):
                    cm_resized[i, j] = cm[i, j]
            cm = cm_resized
        sensitivity, specificity = calculate_sensitivity_specificity(cm)
        sensibilities.append(sensitivity)
        specificities.append(specificity)
        conf_matrices.append(cm)

    # Agora, calcule as métricas totais com todos os dados acumulados
    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_cm = confusion_matrix(all_labels, all_preds)
    overall_sensitivity, overall_specificity = calculate_sensitivity_specificity(overall_cm)


    return (
        overall_accuracy,
        overall_sensitivity,
        overall_specificity,
        conf_matrices,
        sensibilities,
        specificities,
        model,
    )


import random


def generate_random_seeds(size=1, seed_range=(0, 2**32 - 1)):
    return [random.randint(seed_range[0], seed_range[1]) for _ in range(size)]


# ====================
# 4. GRID SEARCH PARA HIPERPARÂMETROS
# ====================
def save_best_params_to_file(
    file_path, params, test_accuracies, train_accuracies, mean_accuracy_test,
    sensibilities, specificities
):
    #result_line = f"mean accuracy test:{mean_accuracy_test}\ntest Accuracy: {test_accuracies}\ntrain Accuracy: {train_accuracies}\n Sensitivity: {sensibilities}\n Specificity: {specificities}\n Params: {params}\n"
    result_line = (
        f"Mean Accuracy Test: {mean_accuracy_test:.4f}\n"
        f"Test Accuracy: {test_accuracies}\n"
        f"Train Accuracy: {train_accuracies}\n"
        f"Sensitivity: {sensibilities}\n"
        f"Specificity: {specificities}\n"
        f"Mean Sensitivity: {np.mean(sensibilities):.4f}\n"
        f"Mean Specificity: {np.mean(specificities):.4f}\n"
        f"Params: {params}\n"
    )

    with open(file_path, "w") as file:
        file.write(result_line)


def grid_search():
    learning_rates = [1e-3]
    batch_sizes = [55]
    epochs_list = [80]
    input_sizes_and_convolucao = {(128, 128): (128, 128, 3)}
    dropout_rates = [0.5]
    best_params = {}
    seed_vector = [193325074]#generate_random_seeds()
    path_result = "melhores.txt"
    best_accuracy = 0
    best_sensitivity = 0
    best_specificity = 0
    best_model_score = -float('inf')  # Inicializa com um valor muito baixo
    for (
        lr,
        batch_size,
        epochs,
        (input_size, convolucao),
        dropout,
        seed,
    ) in itertools.product(
        learning_rates,
        batch_sizes,
        epochs_list,
        input_sizes_and_convolucao.items(),
        dropout_rates,
        seed_vector,
    ):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(
            f"Testando: LR={lr}, Batch size={batch_size}, Epochs={epochs}, Input size={input_size}, Dropout={dropout}, seed={seed}"
        )
        start_time = time.time()

        # Chamada de 'train_and_evaluate' que agora retorna as métricas gerais
        overall_accuracy, overall_sensitivity, overall_specificity, conf_matrices, sensibilities, specificities, model = train_and_evaluate(
            input_size, lr, convolucao, dropout, batch_size, epochs
        )
        elapsed_time = time.time() - start_time
        # Não é mais necessário calcular a média, pois já temos as métricas totais
        print(
            f"Resultado: Acurácia={overall_accuracy:.4f}, Sensibilidade={overall_sensitivity:.4f}, Especificidade={overall_specificity:.4f}")

        elapsed_time = time.time() - start_time  # Tempo de execução total
        print(f"Tempo total: {elapsed_time:.2f}s")
        params = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "input_size": input_size,
            "dropout": dropout,
            "seed": seed,
        }

        if is_better_model(overall_sensitivity, overall_specificity, overall_accuracy, best_model_score):
            salvaGeral(
                overall_accuracy,
                params,
                model,
                #test_accuracies, AQUI GPT" CMO GUARDAR ISSO?
                #train_accuracy, AQUI GPT" CMO GUARDAR ISSO?
                conf_matrices,
                sensibilities,
                specificities,
                path_result,
            )

            # Atualiza as melhores métricas e parâmetros
            best_accuracy = overall_accuracy
            best_sensitivity = overall_sensitivity
            best_specificity = overall_specificity
            best_model_score = 0.4 * best_sensitivity + 0.1 * best_specificity + 0.5 * best_accuracy
            best_params = params

    # Imprime as melhores configurações
    print("\nMelhor Configuração:")
    print(f"Acurácia: {best_accuracy:.4f}")
    print(f"Sensibilidade: {best_sensitivity:.4f}")
    print(f"Especificidade: {best_specificity:.4f}")
    print(f"Hiperparâmetros: {best_params}")


def salvaGeral(mean_accuracy, params, model, test_accuracies, train_accuracy, conf_matrices, sensibilities,
               specificities, path_result):
    # Salva os melhores parâmetros no arquivo
    save_best_params_to_file(
        path_result, params, test_accuracies, train_accuracy, mean_accuracy, sensibilities, specificities
    )

    # Salva as matrizes de confusão
    save_confusion_matrices_to_txt(conf_matrices)

    # Salva o modelo com um nome mais dinâmico (pode incluir parâmetros)
    model_filename = "MobileNet_.h5"
    model.save(model_filename)
    print(f"Modelo salvo como {model_filename}")


def save_confusion_matrices_to_txt(conf_matrices, output_file="confusion_matrices.txt"):
    with open(output_file, "w") as file:
        for i, cm in enumerate(conf_matrices, start=1):
            file.write(f"Fold {i}:\n")
            if cm.shape != (2, 2):
                cm_resized = np.zeros((2, 2), dtype=int)
                for i in range(min(2, cm.shape[0])):
                    for j in range(min(2, cm.shape[1])):
                        cm_resized[i, j] = cm[i, j]
                cm = cm_resized
            np.savetxt(file, cm, fmt="%d", delimiter=",")
            file.write("\n")
    print(f"Confusion matrices saved to {output_file}")


def load_confusion_matrices(file_path="confusion_matrices.txt"):
    matrices = []
    with open(file_path, "r") as file:
        current_matrix = []
        for line in file:
            line = line.strip()
            if line.startswith("Fold"):
                if current_matrix:
                    matrices.append(np.array(current_matrix, dtype=int))
                    current_matrix = []
            elif line:
                current_matrix.append(list(map(int, line.split(","))))
        if current_matrix:
            matrices.append(np.array(current_matrix, dtype=int))
    return matrices


if __name__ == "__main__":
    grid_search()
