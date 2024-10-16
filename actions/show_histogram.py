import matplotlib.pyplot as plt

# Função para exibir o histograma de uma imagem
def show_histogram(image_pil):
    image_np = image_pil.convert("L")  # Converte para tons de cinza
    hist_data = image_np.histogram()

    plt.figure(figsize=(6, 6))
    plt.bar(range(256), hist_data[:256], color='gray')
    plt.title("Histograma de tons de cinza")
    plt.xlabel("Intensidade de pixel")
    plt.ylabel("Frequência")
    plt.show()