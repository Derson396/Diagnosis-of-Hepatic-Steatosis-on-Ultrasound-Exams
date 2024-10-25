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
