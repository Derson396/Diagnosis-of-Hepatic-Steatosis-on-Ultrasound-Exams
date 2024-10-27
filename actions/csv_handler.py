import os
import csv

def ensure_directory_exists(directory):
    """Cria o diretório se ele não existir."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_roi_data_to_csv(file_path, data):
    """Escreve os dados das ROIs no arquivo CSV."""
    # Verificar se o diretório existe e criá-lo se necessário
    ensure_directory_exists(os.path.dirname(file_path))

    # Verificar se o arquivo CSV já existe
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Se o arquivo não existe, escrever o cabeçalho
        if not file_exists:
            writer.writerow(['Arquivo', 'Classe', 'Canto Superior Esquerdo Fígado', 'Canto Superior Esquerdo Rim', 'HI'])

        # Escrever os dados
        writer.writerow(data)

def write_texture_features_to_csv(file_path, features):
    """Escreve as características de textura no arquivo CSV."""
    # Verificar se o diretório existe e criá-lo se necessário
    ensure_directory_exists(os.path.dirname(file_path))

    # Verificar se o arquivo CSV já existe
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Se o arquivo não existe, escrever o cabeçalho
        if not file_exists:
            writer.writerow(['Arquivo', 'Classe', 'Coarseness', 'Contrast', 'Periodicity', 'Roughness',
                             'glcm_distance_1_contrast','glcm_distance_1_dissimilarity','glcm_distance_1_homogeneity','glcm_distance_1_energy','glcm_distance_1_correlation','glcm_distance_1_ASM','glcm_distance_1_entropy',
                             'glcm_distance_2_contrast','glcm_distance_2_dissimilarity','glcm_distance_2_homogeneity','glcm_distance_2_energy','glcm_distance_2_correlation','glcm_distance_2_ASM','glcm_distance_2_entropy',
                             'glcm_distance_4_contrast','glcm_distance_4_dissimilarity','glcm_distance_4_homogeneity','glcm_distance_4_energy','glcm_distance_4_correlation','glcm_distance_4_ASM','glcm_distance_4_entropy',
                             'glcm_distance_8_contrast','glcm_distance_8_dissimilarity','glcm_distance_8_homogeneity','glcm_distance_8_energy','glcm_distance_8_correlation','glcm_distance_8_ASM','glcm_distance_8_entropy',
                             ])

        # Escrever os dados das características de textura
        writer.writerow(features)
