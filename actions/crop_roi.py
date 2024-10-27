from PIL import Image

def crop_roi(image_pil, x1, y1, x2, y2, canvas_width, canvas_height):
    """Recorta a ROI ajustando as coordenadas conforme o redimensionamento da imagem."""
    # Tamanho da imagem original
    original_width, original_height = image_pil.size

    # Ajustar as coordenadas conforme a proporção entre canvas e a imagem original
    scale_x = original_width / canvas_width
    scale_y = original_height / canvas_height

    # Ajustar as coordenadas da ROI de acordo com a escala
    x1_scaled = int(x1 * scale_x)
    y1_scaled = int(y1 * scale_y)
    x2_scaled = int(x2 * scale_x)
    y2_scaled = int(y2 * scale_y)

    # Recortar a ROI ajustada na imagem original
    roi = image_pil.crop((x1_scaled, y1_scaled, x2_scaled, y2_scaled))
    return roi
