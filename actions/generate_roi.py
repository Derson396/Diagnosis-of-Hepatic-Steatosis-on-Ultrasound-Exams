import numpy as np
from PIL import Image
import os

def process_rois(roi_rim, roi_figado, output_path):
    """Processa as duas ROIs (rim e fígado), calcula o HI e normaliza a ROI do fígado.
    
    Args:
        roi_rim (PIL.Image): Imagem da ROI do rim.
        roi_figado (PIL.Image): Imagem da ROI do fígado.
        output_path (str): Caminho para salvar a imagem do fígado normalizada.
    
    Returns:
        str: Caminho da imagem normalizada do fígado.
        float: Índice hepatorenal (HI).
    """
    # Converter as ROIs para arrays numpy
    rim_array = np.array(roi_rim.convert("L"))  # Converter para tons de cinza
    figado_array = np.array(roi_figado.convert("L"))  # Converter para tons de cinza

    # Calcular as médias dos tons de cinza
    rim_mean = np.mean(rim_array)
    figado_mean = np.mean(figado_array)

    # Calcular o Índice Hepatorenal (HI)
    if rim_mean == 0:
        raise ValueError("A média dos tons de cinza da ROI do rim é zero. Não é possível calcular o HI.")
    
    hi = figado_mean / rim_mean

    # Normalizar os tons de cinza da ROI do fígado
    figado_normalizado_array = figado_array * hi
    figado_normalizado_array = np.clip(figado_normalizado_array, 0, 255)  # Garantir que os valores estejam no intervalo válido

    # Criar a nova imagem normalizada
    figado_normalizado = Image.fromarray(figado_normalizado_array.astype(np.uint8))

    # Salvar a imagem normalizada
    figado_normalizado.save(output_path)

    # Retornar o caminho da nova ROI do fígado normalizada e o HI
    return output_path, hi
