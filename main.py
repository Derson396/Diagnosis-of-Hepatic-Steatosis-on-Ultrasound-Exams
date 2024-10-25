import os  # operating system functionalities
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import numpy as np
from actions.generate_roi import process_rois
from actions.load_image import load_dataset, load_image
from actions.crop_roi import crop_roi
from actions.show_histogram import show_histogram
from actions.compute_glcm import compute_glcm, show_glcm_heatmap
from RoiApp import RoiApp
from tkinter import messagebox


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Imagens")
        self.images = load_dataset()

        # Configurar a interface
        self.canvas = tk.Canvas(root, width=600, height=600, bg="gray")
        self.canvas.grid(row=0, column=0, columnspan=4)

        # Variáveis para armazenar a imagem atual e a ROI
        self.current_image_pil = None
        self.roi_coords = None
        self.roiRim_coords = None
        self.roiFigado_coords = None
        self.rect = None
        self.roi_app = None
        self.roi_count = 0  # Contador de ROIs

        # Comboboxes para selecionar paciente e imagem
        self.paciente_var = tk.StringVar()
        self.imagem_var = tk.StringVar()

        pacientes = [str(i) for i in range(len(self.images[0]))]
        imagens = [str(i) for i in range(len(self.images[0][0]))]

        self.combo_paciente = ttk.Combobox(
            root, textvariable=self.paciente_var, values=pacientes, state="readonly"
        )
        self.combo_paciente.grid(row=1, column=0)
        self.combo_paciente.set("Selecione Paciente")

        self.combo_imagem = ttk.Combobox(
            root, textvariable=self.imagem_var, values=imagens, state="readonly"
        )
        self.combo_imagem.grid(row=1, column=1)
        self.combo_imagem.set("Selecione Imagem")

        btn_load_image_arquivo = ttk.Button(
            root,
            text="Carregar Imagem por um arquivo",
            command=self.load_imagem_arquivo,
        )
        btn_load_image_arquivo.grid(row=2, column=0)

        # Botões de controle
        btn_load_image = ttk.Button(
            root, text="Carregar Imagem", command=self.load_image
        )
        btn_load_image.grid(row=2, column=1)

        btn_crop_roi = ttk.Button(
            root, text="Recortar ROI", command=self.start_crop_roi
        )
        btn_crop_roi.grid(row=2, column=2)

        """         btn_stop_crop_roi = ttk.Button(
            root, text="Parar de Recortar ROI", command=self.stop_crop_roi
        )
        btn_stop_crop_roi.grid(row=2, column=3) """

        btn_generate__roi = ttk.Button(
            root, text="Gerar ROI", command=self.start_generate_two_rois
        )
        btn_generate__roi.grid(row=2, column=3)

        """     btn_show_histogram = ttk.Button(
            root, text="Exibir Histograma", command=self.show_histogram
        )
        btn_show_histogram.grid(row=2, column=5)

        btn_compute_glcm = ttk.Button(
            root, text="Calcular GLCM", command=self.compute_glcm
        )
        btn_compute_glcm.grid(row=2, column=6) """

        btn_exit = ttk.Button(root, text="Sair", command=root.quit)
        btn_exit.grid(row=2, column=7)

        self.selection_status_label = tk.Label(root, font=("Arial", 12))
        self.selection_status_label.grid(row=3, column=0, columnspan=4, pady=10)
        self.selection_status_label.grid_remove()  # Esconder inicialmente

    def load_imagem_arquivo(self):
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem",
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp")],
        )
        if file_path:
            self.current_image_pil = Image.open(file_path)
            self.display_image(self.current_image_pil)

    def load_image(self):
        """Carrega a imagem selecionada com base no paciente e índice de imagem."""
        try:
            self.paciente = int(self.paciente_var.get())
            self.image_idx = int(self.imagem_var.get())
        except ValueError:
            print("Selecione um paciente e uma imagem válidos.")
            return

        # Carregar a imagem
        try:
            self.current_image_pil = load_image(
                self.images, self.paciente, self.image_idx
            )
            self.display_image(self.current_image_pil)
        except ValueError as e:
            print(e)

    def display_image(self, image_pil):
        """Exibe a imagem no canvas."""
        self.tk_image = ImageTk.PhotoImage(image=image_pil.resize((600, 600)))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def start_crop_roi(self):
        """Inicia o processo de recorte da ROI."""
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)

    def stop_crop_roi(self):
        """Para o processo de recorte da ROI."""
        self.canvas.unbind("<ButtonPress-1>")

    def on_button_press(self, event):
        """Quando o botão do mouse é pressionado, desenhar um quadrado fixo de 28x28 com o mouse no centro."""
        # Coordenadas do centro (onde o mouse foi clicado)
        center_x, center_y = event.x, event.y

        # Calcular os cantos do quadrado 28x28
        x1 = center_x - 14
        y1 = center_y - 14
        x2 = center_x + 14
        y2 = center_y + 14
        self.roi_coords = [(x1, y1), (x2, y2)]

        # Desenhar o quadrado no canvas
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline="green")
        # Recortar e exibir a ROI automaticamente
        self.crop_and_show_roi()

    def crop_and_show_roi(self):
        """Recortar a ROI e exibi-la."""
        x1, y1 = self.roi_coords[0]
        x2, y2 = self.roi_coords[1]

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        roi = crop_roi(
            self.current_image_pil, x1, y1, x2, y2, canvas_width, canvas_height
        )
        new_window = tk.Toplevel(self.root)
        if self.roi_app:
            self.roi_app.root.destroy()
        self.roi_app = RoiApp(new_window, roi, self.image_idx, self.paciente)
        self.stop_crop_roi()

    def show_histogram(self):
        """Exibir o histograma da imagem ou ROI atual."""
        if self.current_image_pil is not None:
            show_histogram(self.current_image_pil)

    def compute_glcm(self):
        """Computa a GLCM e exibe os descritores de textura e a matriz de co-ocorrência da ROI atual."""
        if self.current_image_pil is not None:
            try:
                # Converte a imagem para array NumPy
                image_array = np.array(
                    self.current_image_pil.convert("L")
                )  # Convertendo para tons de cinza

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
    
    def start_generate_two_rois(self):
        """Inicia o processo para selecionar duas ROIs e atualiza o status."""
        self.roi_coords = []  # Reiniciar coordenadas
        self.roiRim_coords = None
        self.roiFigado_coords = None
        self.rects = []  # Reiniciar os retângulos desenhados
        self.roi_count = 0  # Reiniciar contador de ROIs
        self.canvas.bind("<ButtonPress-1>", self.on_button_press_two_rois)
        
        # Atualizar o status para informar que o primeiro ROI é do RIM
        self.selection_status_label.config(text="Selecione a região do rim.")
        self.selection_status_label.grid()

    def on_button_press_two_rois(self, event):
        """Captura duas ROIs de 28x28 ao clicar com o mouse."""
        if self.roi_count < 2:
            # Coordenadas do centro (onde o mouse foi clicado)
            center_x, center_y = event.x, event.y

            # Calcular os cantos do quadrado 28x28
            x1 = center_x - 14
            y1 = center_y - 14
            x2 = center_x + 14
            y2 = center_y + 14
            if self.roi_count == 0:
                self.roiRim_coords = [(x1, y1), (x2, y2)]
                self.selection_status_label.config(text="Agora selecione a região do fígado.")
            elif self.roi_count == 1:
                self.roiFigado_coords = [(x1, y1), (x2, y2)]
            
            # Desenhar o quadrado no canvas
            rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline="green")
            self.rects.append(rect)

            self.roi_count += 1

            if self.roi_count == 2:
                # Quando as duas ROIs forem selecionadas, chamar o processamento
                self.selection_status_label.config(text="Processando as ROIs...")
                self.process_two_rois()

    def process_two_rois(self):
        """Processa as duas ROIs selecionadas."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        # Exemplo de processamento: calcular a sobreposição ou diferenças
        x1_R, y1_R = self.roiRim_coords[0]
        x2_R, y2_R = self.roiRim_coords[1]

        roiRim = crop_roi(
            self.current_image_pil, x1_R, y1_R, x2_R, y2_R, canvas_width, canvas_height
        )

        x1_F, y1_F = self.roiFigado_coords[0]
        x2_F, y2_F = self.roiFigado_coords[1]

        roiFigado = crop_roi(
            self.current_image_pil, x1_F, y1_F, x2_F, y2_F, canvas_width, canvas_height
        )

        output_path, hi = process_rois(roiRim, roiFigado, f"Rois/ROI_{self.paciente}_{self.image_idx}.png")
            # Mostrar mensagem com o HI e abrir a imagem normalizada do fígado
        messagebox.showinfo(
                "Índice Hepatorenal Calculado",
                f"Índice Hepatorenal (HI): {hi:.2f}\nImagem do fígado normalizada salva em: {output_path}"
            )

        if self.roi_app:
            self.roi_app.root.destroy()
        normalized_image = Image.open(output_path)
        new_window = tk.Toplevel(self.root)
        self.roi_app = RoiApp(new_window, normalized_image, self.image_idx, self.paciente)
        self.selection_status_label.grid_remove()

        # Atualizar o status de que as ROIs foram processadas
        self.selection_status_label.config(text="Seleção de ROIs finalizada. Processo concluído!")

        # deletar os retângulos
        for rect in self.rects:
            self.canvas.delete(rect)


# Iniciar a aplicação
root = tk.Tk()
app = ImageApp(root)
root.mainloop()
