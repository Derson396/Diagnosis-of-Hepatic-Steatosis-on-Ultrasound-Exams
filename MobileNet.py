import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dropout
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold


def mobileNet(
    input_size=(32, 32), learning_rate=1e-4, convolucao=(32, 32, 1), dropout=0
):
    # ====================
    # CONFIGURÁVEL: PATHS
    # ====================
    csv_path = "roi_data.csv"
    image_dir = "Rois/Normalizadas/"

    # ====================
    # CONFIGURÁVEL: HIPERPARÂMETROS
    # ====================
    input_size = input_size  # Tamanho das imagens para MobileNet
    batch_size = 32  # Tamanho do lote
    epochs = 40  # Número de épocas
    learning_rate = learning_rate  # Taxa de aprendizado
    k_folds = 55  # Número de divisões na validação cruzada

    # ====================
    # 1. CARREGAMENTO DOS DADOS
    # ====================
    # Lê o CSV
    data = pd.read_csv(csv_path)

    # Extrair o ID do paciente a partir do nome do arquivo
    data["Paciente"] = (
        data["Arquivo"].str.extract(r"ROI_(\d+)_\d+", expand=False).astype(int)
    )
    # Obter a lista única de pacientes
    pacientes = data["Paciente"].unique()
    # ====================
    # 4. VALIDAÇÃO CRUZADA
    # ====================
    accuracies = []
    conf_matrices = []
    train_accuracies = []
    val_accuracies = []

    # Dividir os pacientes para validação cruzada
    for fold, test_patient_id in enumerate(pacientes, 1):
        print(f"\nFold {fold}/{len(pacientes)}")

        # Separar os dados de treino e teste baseando-se no paciente
        test_data = data[data["Paciente"] == test_patient_id]
        train_data = data[data["Paciente"] != test_patient_id]

        # Pré-processamento dos dados
        x_train = np.array(
            [
                load_and_preprocess_image(f, input_size, image_dir)
                for f in train_data["Arquivo"]
            ]
        )
        y_train = train_data["Classe"].values
        x_test = np.array(
            [
                load_and_preprocess_image(f, input_size, image_dir)
                for f in test_data["Arquivo"]
            ]
        )
        y_test = test_data["Classe"].values
        # Criar e treinar o modelo
        model = create_mobilenet_model(
            num_classes=2,
            learning_rate=learning_rate,
            convolucao=convolucao,
            dropout=dropout,
        )

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )

        # Avaliar o modelo
        preds = model.predict(x_test, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        conf_matrices.append(confusion_matrix(y_test, y_pred))

        # Salvar acurácia de treino e validação
        train_accuracies.append(history.history["accuracy"])
        val_accuracies.append(history.history["val_accuracy"])

    # ====================
    # 5. PLOTAR MÉTRICAS
    # ====================
    # Plotar a acurácia média por fold
    """plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(pacientes) + 1), accuracies, marker="o")
    plt.title("Acurácia por Fold")
    plt.xlabel("Fold")
    plt.ylabel("Acurácia")
    plt.grid()
    plt.show()

    # Plotar gráficos de aprendizado (treinamento e validação)
    plt.figure(figsize=(10, 5))
    plt.plot(
        np.mean(train_accuracies, axis=0), label="Acurácia de Treinamento", color="blue"
    )
    plt.plot(np.mean(val_accuracies, axis=0), label="Acurácia de Validação", color="red")
    plt.title("Gráfico de Aprendizado (Acurácia de Treinamento e Validação)")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid()
    plt.show()

    # Plotar matrizes de confusão para cada fold
    for i, cm in enumerate(conf_matrices):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusão - Fold {i + 1}")
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.show()"""

    # Resultados finais
    # print(f"Acurácia média: {np.mean(accuracies):.4f}")
    # print(f"Desvio padrão: {np.std(accuracies):.4f}")
    return accuracies


# ====================
# 2. DEFINIR O MODELO COM FINE-TUNING
# ====================
def create_mobilenet_model(num_classes, learning_rate, convolucao, dropout):
    base_model = MobileNet(
        weights="imagenet", include_top=False, input_shape=convolucao
    )
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(dropout)(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)

    # Congela as camadas base inicialmente para fine-tuning
    for layer in base_model.layers:
        layer.trainable = False

    # Compilar o modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ====================
# 3. PREPROCESSAMENTO
# ====================
def load_and_preprocess_image(file_path, target_size, image_dir):
    file_path = os.path.join(image_dir, file_path)
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = image / 255.0  # Normalização para o intervalo [0, 1]
    return image


import itertools


def tune_hyperparameters():
    # Defina os hiperparâmetros para testar
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-7, 1e-8]
    batch_sizes = [16, 32, 20]
    epochs_list = [20, 40, 60, 80, 100]
    # Mapeamento de tamanhos de entrada para formas específicas (altura, largura, canais)
    """input_sizes_and_convolucao = {
        (32, 32): (32, 32, 3),
        (64, 64): (64, 64, 3), 
        (224, 224): (224, 224, 3), 
    }"""
    input_sizes_and_convolucao = {
        (128, 128): (128, 128, 3),
        (160, 160): (160, 160, 3),
        (192, 192): (192, 192, 3),
        (224, 224): (224, 224, 3),
    }
    dropout = [0, 0.1, 0.2, 0.5, 0.7, 0.8]

    best_accuracy = 0
    best_params = {}
    Totalaccuracies = []
    # Realizar a busca por combinação de hiperparâmetros
    for lr, batch_size, epochs, (input_size, convolucao), dropout in itertools.product(
        learning_rates,
        batch_sizes,
        epochs_list,
        input_sizes_and_convolucao.items(),
        dropout,
    ):
        print(
            f"\n\n\tTestando: LR={lr}, Batch size={batch_size}, Epochs={epochs}, Input size={input_size}, dropout:{dropout}"
        )

        # Chama a função mobileNet com os parâmetros ajustados
        print("\n\tIniciando treinamento com esses hiperparâmetros...\n\n")
        accuracies = mobileNet(
            input_size=input_size,
            learning_rate=lr,
            convolucao=convolucao,
            dropout=dropout,
        )

        # Avaliar a acurácia média no final da validação cruzada
        mean_accuracy = np.mean(accuracies)
        Totalaccuracies.append(accuracies)
        # Armazenar os melhores resultados
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = {
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "input_size": input_size,
                "dropout": dropout,
            }

    print(f"\nMelhor Acurácia: {best_accuracy:.4f}")
    print(f"Melhores Hiperparâmetros: {best_params}")

    print(f"\n\n\tTodas as acuracias:\n{Totalaccuracies}")


# Chama a função para testar diferentes combinações de hiperparâmetros
tune_hyperparameters()
