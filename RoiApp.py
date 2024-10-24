import os # operating system functionalities
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import ImageTk
from actions.show_histogram import show_histogram
from actions.compute_glcm import compute_glcm, show_glcm_heatmap
from tkinter import messagebox

class RoiApp:
    def __init__(self, root,imagem,imagem_var,paciente_var):
        self.root = root
        self.root.title("Visualizador de ROIs")
        # Configurar a interface
        self.canvas = tk.Canvas(root, width=600, height=600, bg="gray")
        self.canvas.grid(row=0, column=0, columnspan=4)
        
        self.current_image_pil = imagem
        self.image_var = imagem_var
        self.pacient_var = paciente_var
        
        btn_save_kidney = ttk.Button(root, text="Salvar como Rim", command=self.save_kidney)
        btn_save_kidney.grid(row=2, column=0)
        
        btn_save_liver = ttk.Button(root, text="Salvar como Figado", command=self.save_liver)
        btn_save_liver.grid(row=2, column=1)
        
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

    def save_kidney(self):
        try:
            pacient_int = int(self.pacient_var)  # Converte para inteiro para comparação
            image_int = int(self.image_var)  # Converte para inteiro, se necessário

            # Cria o diretório se não existir
            save_directory = "Rois/Figado"
            os.makedirs(save_directory, exist_ok=True)  # Cria o diretório se não existir

            # Define o caminho do arquivo
            if 0 < pacient_int < 17:
                file_path = os.path.join(save_directory, f"SaudavelPacient{pacient_int}Imagem{image_int}.png")
            else:
                file_path = os.path.join(save_directory, f"DoentePacient{pacient_int}Imagem{image_int}.png")

            # Salva a imagem
            new_size = (600, 600)  # Define o novo tamanho
            resized_image = self.current_image_pil.resize(new_size)
            resized_image.save(file_path)
        except Exception as e:
            print(f"Erro ao salvar a imagem: {e}")
            
    def save_liver(self):
        try:
            print
            pacient_int = int(self.pacient_var)  # Converte para inteiro para comparação
            image_int = int(self.image_var)  # Converte para inteiro, se necessário

            # Cria o diretório se não existir
            save_directory = "Rois/Rim"
            os.makedirs(save_directory, exist_ok=True)  # Cria o diretório se não existir

            # Define o caminho do arquivo
            if 0 < pacient_int < 17:
                file_path = os.path.join(save_directory, f"SaudavelPacient{pacient_int}Imagem{image_int}.png")
            else:
                file_path = os.path.join(save_directory, f"DoentePacient{pacient_int}Imagem{image_int}.png")

            # Salva a imagem
            new_size = (600, 600)  # Define o novo tamanho
            resized_image = self.current_image_pil.resize(new_size)
            resized_image.save(file_path)
        except Exception as e:
            print(f"Erro ao salvar a imagem: {e}")
            