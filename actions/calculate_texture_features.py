import numpy as np
from skimage import filters

def calculate_texture_features(image):
    # Normalizar a imagem
    img = np.array(image)

    # Normalizar a imagem
    img = img.astype(np.float32) / 255.0

    # Coarseness
    # Usando um operador de diferença em múltiplas escalas
    scales = [1, 2, 4, 8]
    coarseness = 0
    for scale in scales:
        shift_img = np.roll(img, scale, axis=0)  # Shift para baixo
        diff = np.abs(img - shift_img)
        coarseness += np.mean(diff)
    coarseness /= len(scales)  # Média entre todas as escalas

    # Contrast
    # Usar desvio padrão para capturar variação
    contrast = np.std(img)

    # Periodicity
    # Usando FFT para capturar o pico da frequência dominante
    f_transform = np.fft.fft2(img)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    
    # Encontra a frequência dominante (maior valor no espectro)
    periodicity = np.max(magnitude_spectrum)

    # Roughness
    # Usar diferença de pixels adjacentes
    roughness = np.mean(np.abs(img[:-1, :-1] - img[1:, :-1])) + \
                np.mean(np.abs(img[:-1, :-1] - img[:-1, 1:]))
    return {
        'Coarseness': coarseness,
        'Contrast': contrast,
        'Periodicity': periodicity,
        'Roughness': roughness
    }
