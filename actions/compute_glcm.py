import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops  
from tkinter import messagebox

def get_angles(start=0,end=360,step=1):
    return np.arange(start, end, step)

def compute_glcm(image, distances=[1,2,4,8], angles=get_angles(), levels=256):
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
    glcm_radial = np.mean(glcm, axis=3)

    features = extract_features_from_glcm(glcm_radial)

    return glcm_radial, features

def extract_features_from_glcm(radial_glcms):
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    radial_glcms_4d = np.expand_dims(radial_glcms, axis=-1)
    features = {prop: [] for prop in properties}
    for prop in properties:
        prop_value = graycoprops(radial_glcms_4d, prop).flatten()
        features[prop].append(prop_value)
    return features


def show_features(features,properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],distances=[1,2,4,8]):
    for i, distance in enumerate(distances):
        features_str=f'\n (Distância {distance}):'
        for prop in properties:
            values = features[prop][0]
            features_str += f"\n {prop.capitalize()} : {values[i]}"
        messagebox.showinfo("Descritores de Textura (GLCM)", features_str)

def show_glcm_heatmap(radial_glcms,distances=[1,2,4,8]):
    num_matrices = radial_glcms.shape[2]  # Número de matrizes de GLCM

    # Para cada matriz, exibir o heatmap
    for i in range(num_matrices):
        plt.figure(figsize=(6, 6))
        plt.imshow(radial_glcms[:, :, i], cmap='hot', interpolation='nearest')  # Ajustado para 3D
        plt.title(f"Heatmap da GLCM (Distancia {distances[i]})")
        plt.colorbar(label='Intensidade')
        plt.show()