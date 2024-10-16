import os # operating system functionalities
import scipy.io
from pathlib import Path
import numpy as np
from PIL import Image

# Função para carregar o dataset
def load_dataset():
    path_input_dir = Path("data")
    path_data = path_input_dir / "dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"
    data = scipy.io.loadmat(path_data)
    data_array = data['data']
    return data_array['images']

# Função para carregar uma imagem específica
def load_image(images, paciente_idx, image_idx):
    try:
        image = images[0][paciente_idx][image_idx]
        return Image.fromarray(image)
    except IndexError:
        raise ValueError("Índices de paciente ou imagem inválidos.")
