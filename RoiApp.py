import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from actions.show_histogram import show_histogram
from actions.compute_glcm import compute_glcm, show_glcm_heatmap, show_features
from tkinter import messagebox
import os

class RoiApp:
    def __init__(self, root, imagem, imagem_var, paciente_var):
        self.root = root
        self.root.title("Visualizador de ROIs")

        # Configurar a interface
        self.canvas = tk.Canvas(root, width=600, height=600, bg="gray")
        self.canvas.grid(row=0, column=0, columnspan=5)

        self.current_image_pil = imagem
        self.image_var = imagem_var
        self.pacient_var = paciente_var

        self.zoom_level = max(1.0, self.calculate_initial_zoom() * 1.5)
        self.pan_x, self.pan_y = self.calculate_initial_pan()

        # Botões de zoom
        btn_zoom_in = ttk.Button(root, text="Zoom +", command=self.zoom_in)
        btn_zoom_in.grid(row=1, column=0)

        btn_zoom_out = ttk.Button(root, text="Zoom -", command=self.zoom_out)
        btn_zoom_out.grid(row=1, column=1)

        # Botões de controle
        btn_save_kidney = ttk.Button(root, text="Salvar como Rim", command=self.save_kidney)
        btn_save_kidney.grid(row=2, column=0)

        btn_save_liver = ttk.Button(root, text="Salvar como Fígado", command=self.save_liver)
        btn_save_liver.grid(row=2, column=1)

        btn_show_histogram = ttk.Button(root, text="Exibir Histograma", command=self.show_histogram)
        btn_show_histogram.grid(row=2, column=2)

        btn_compute_glcm = ttk.Button(root, text="Calcular GLCM", command=self.compute_glcm)
        btn_compute_glcm.grid(row=2, column=3)

        btn_exit = ttk.Button(root, text="Sair", command=root.destroy)
        btn_exit.grid(row=2, column=4)

        # Inicializar e exibir imagem com zoom e pan iniciais
        self.update_display()

        # Eventos de pan
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)

        self.zoom_out()
        self.zoom_out()

    def calculate_initial_zoom(self):
        canvas_width, canvas_height = 600, 600
        img_width, img_height = self.current_image_pil.size
        print(self.current_image_pil.size)
        zoom_w = canvas_width / img_width
        zoom_h = canvas_height / img_height
        return min(zoom_w, zoom_h)

    def calculate_initial_pan(self):
        canvas_width, canvas_height = 600, 600
        img_width = int(self.current_image_pil.width * self.zoom_level)
        img_height = int(self.current_image_pil.height * self.zoom_level)
        pan_x = (canvas_width - img_width) // 2
        pan_y = (canvas_height - img_height) // 2
        return pan_x, pan_y

    def zoom_in(self):
        self.zoom_level *= 1.2  # Aumenta o zoom em 20%
        self.update_display()

    def zoom_out(self):
        self.zoom_level /= 1.2  # Diminui o zoom em 20%
        self.update_display()

    def update_display(self):
        print(self.zoom_level)
        if self.current_image_pil:
            # Redimensiona a imagem com o nível de zoom atual
            zoomed_image = self.current_image_pil.resize(
                (int(self.current_image_pil.width * self.zoom_level),
                 int(self.current_image_pil.height * self.zoom_level)),
                Image.LANCZOS
            )
            self.tk_image = ImageTk.PhotoImage(zoomed_image)
            self.canvas.delete("all")
            # Centraliza a imagem
            img_x = max(0, (600 - zoomed_image.width) // 2 + self.pan_x)
            img_y = max(0, (600 - zoomed_image.height) // 2 + self.pan_y)
            self.canvas.create_image(img_x, img_y, anchor=tk.NW, image=self.tk_image)

    def start_pan(self, event):
        """Inicia o pan da imagem."""
        # Registra a posição inicial do pan
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def do_pan(self, event):
        """Realiza o pan da imagem."""
        # Calcula a diferença de movimento
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y

        # Atualiza as coordenadas de pan com base no movimento
        self.pan_x += dx
        self.pan_y += dy

        # Atualiza a posição inicial do pan para a nova posição do evento
        self.pan_start_x = event.x
        self.pan_start_y = event.y

        # Atualiza a exibição com as novas coordenadas de pan
        self.update_display()

    def show_histogram(self):
        if self.current_image_pil:
            show_histogram(self.current_image_pil)

    def compute_glcm(self):
        if self.current_image_pil:
            try:
                image_array = np.array(self.current_image_pil.convert("L"))
                glcm, features = compute_glcm(image_array)
                show_glcm_heatmap(glcm)
                show_features(features)
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
            os.makedirs(
                save_directory, exist_ok=True
            )  # Cria o diretório se não existir

            if self.paciente != None:
                classe = 0 if self.paciente <= 16 else 1  # Pacientes 0 a 16 são saudáveis, restante com esteatose
            else:
                classe = input('Preencha a classe (0 para saudavel 1 para doente):')

            # Define o caminho do arquivo
            if self.pacient_var == None or self.image_var == None:
                pacient_int = input("Escreva o numero do paciente :")
                image_int = input("Escreva qual é a imagem do paciente : ")

            # Define o caminho do arquivo
            if classe == 0:
                file_path = os.path.join(
                    save_directory, f"SaudavelPacient{pacient_int}Imagem{image_int}.png"
                )
            else:
                file_path = os.path.join(
                    save_directory, f"DoentePacient{pacient_int}Imagem{image_int}.png"
                )

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
            os.makedirs(
                save_directory, exist_ok=True
            )  # Cria o diretório se não existir

            # Define o caminho do arquivo
            if self.pacient_var == None or self.image_var == None:
                pacient_int = input("Escreva o numero do paciente :")
                image_int = input("Escreva qual é a imagem do paciente : ")

            if 0 < pacient_int < 17:
                file_path = os.path.join(
                    save_directory, f"SaudavelPacient{pacient_int}Imagem{image_int}.png"
                )
            else:
                file_path = os.path.join(
                    save_directory, f"DoentePacient{pacient_int}Imagem{image_int}.png"
                )

            # Salva a imagem
            new_size = (600, 600)  # Define o novo tamanho
            resized_image = self.current_image_pil.resize(new_size)
            resized_image.save(file_path)
        except Exception as e:
            print(f"Erro ao salvar a imagem: {e}")


