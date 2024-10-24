import os # operating system functionalities
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import ImageTk
from actions.show_histogram import show_histogram
from actions.compute_glcm import compute_glcm, show_glcm_heatmap
from tkinter import messagebox

class RoiApp:
    def __init__(self, root,imagem):
        self.root = root
        self.root.title("Visualizador de Imagens e ROIs")
        # Configurar a interface
        self.canvas = tk.Canvas(root, width=600, height=600, bg="gray")
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.current_image_pil = imagem
        btn_show_histogram = ttk.Button(root, text="Exibir Histograma", command=self.show_histogram)
        btn_show_histogram.grid(row=2, column=2)

        btn_compute_glcm = ttk.Button(root, text="Calcular GLCM", command=self.compute_glcm)
        btn_compute_glcm.grid(row=2, column=3)

        btn_exit = ttk.Button(root, text="Sair", command=root.destroy)
        btn_exit.grid(row=2, column=4)
        
        self.tk_image = ImageTk.PhotoImage(image=self.current_image_pil.resize((600, 600)))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        
    def show_histogram(self):
        """Exibir o histograma da imagem ou ROI atual."""
        if self.current_image_pil is not None:
            show_histogram(self.current_image_pil)
    
    def compute_glcm(self):
        """Computa a GLCM e exibe os descritores de textura e a matriz de co-ocorrência da ROI atual."""
        if self.current_image_pil is not None:
            try:
                # Converte a imagem para array NumPy
                image_array = np.array(self.current_image_pil.convert('L'))  # Convertendo para tons de cinza
                
                # Computa a GLCM e os descritores
                glcm, features = compute_glcm(image_array)

                # Exibe o heatmap da GLCM
                show_glcm_heatmap(glcm)

                # Exibe os descritores
                features_str = "\n".join([f"{k}: {v}" for k, v in features.items()])
                messagebox.showinfo("Descritores de Textura (GLCM)", features_str)

            except ValueError as e:
                messagebox.showerror("Erro", str(e))
        else:
            messagebox.showerror("Erro", "Nenhuma ROI carregada para calcular GLCM.")
