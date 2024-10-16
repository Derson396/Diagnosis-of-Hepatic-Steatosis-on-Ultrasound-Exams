from PIL import Image

# Função para recortar uma ROI da imagem
def crop_roi(image_pil, x1, y1, x2, y2):
    # Recorta a ROI na imagem (Image é um objeto PIL)
    roi = image_pil.crop((x1, y1, x2, y2))
    return roi