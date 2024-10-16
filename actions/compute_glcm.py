import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def compute_glcm(image, distances=[1], angles=[0], levels=256):
    """
    Calcula a matriz de co-ocorrência (GLCM) e os descritores de textura.

    Parameters:
    - image: A imagem (ROI) em que a GLCM será calculada (espera uma imagem em tons de cinza).
    - distances: Lista de distâncias para o cálculo da GLCM.
    - angles: Lista de ângulos para o cálculo da GLCM (em radianos).
    - levels: Número de níveis de tons de cinza (padrão 256).

    Returns:
    - glcm: A matriz de co-ocorrência calculada.
    - features: Um dicionário contendo as propriedades de textura calculadas.
    """

    # Verifica se a imagem é em tons de cinza
    if len(image.shape) > 2:
        raise ValueError("A imagem precisa ser em tons de cinza.")

    # Converte a imagem para valores inteiros no intervalo [0, levels-1]
    image = (image / image.max() * (levels - 1)).astype(np.uint8)

    # Calcula a GLCM
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    # Extrai propriedades de textura
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = {prop: graycoprops(glcm, prop).flatten() for prop in properties}

    return glcm, features

def save_glcm_to_file(glcm, filepath="glcm_matrix.txt"):
    """Salva a matriz GLCM em um arquivo .txt."""
    with open(filepath, 'w') as f:
        for i in range(glcm.shape[2]):
            f.write(f"GLCM (Slice {i+1}):\n")
            np.savetxt(f, glcm[:, :, i, 0], fmt='%f')
            f.write("\n")

def show_glcm_heatmap(glcm):
    """
    Exibe o heatmap da matriz de co-ocorrência (GLCM).

    Parameters:
    - glcm: A matriz GLCM a ser exibida (espera uma matriz 4D).
    """
    num_matrices = glcm.shape[2]  # Número de matrizes de GLCM (uma para cada combinação de distância e ângulo)

    # Para cada matriz, exibir o heatmap
    for i in range(num_matrices):
        plt.figure(figsize=(6, 6))
        plt.imshow(glcm[:, :, i, 0], cmap='hot', interpolation='nearest')
        plt.title(f"Heatmap da GLCM - Matriz {i+1}")
        plt.colorbar(label='Intensidade')
        plt.show()