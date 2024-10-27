import os  # operating system functionalities
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import numpy as np
from actions.calculate_texture_features import calculate_texture_features
from actions.generate_roi import process_rois
from actions.load_image import load_dataset, load_image
from actions.crop_roi import crop_roi
from actions.show_histogram import show_histogram
from actions.compute_glcm import compute_glcm, show_glcm_heatmap, show_features
from RoiApp import RoiApp
from tkinter import messagebox
from actions.csv_handler import write_roi_data_to_csv, write_texture_features_to_csv


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Imagens")
        self.images = load_dataset()

        self.zoom_level = 1
        self.pan_x, self.pan_y = 0, 0

        # Configurar a interface
        self.canvas = tk.Canvas(root, bg="gray")
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

        # Botões de zoom
        btn_zoom_in = ttk.Button(root, text="Zoom +", command=self.zoom_in)
        btn_zoom_in.grid(row=1, column=2)

        btn_zoom_out = ttk.Button(root, text="Zoom -", command=self.zoom_out)
        btn_zoom_out.grid(row=1, column=3)

        # Eventos de pan
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)

    def zoom_in(self):
        if self.zoom_level < 4:
            self.zoom_level += 1
            self.update_display()

    def zoom_out(self):
        if self.zoom_level > 1:
            self.zoom_level -= 1
            self.update_display()

    def update_display(self):
        """Atualiza a exibição da imagem no canvas, levando em conta zoom e pan."""
        if self.current_image_pil:
            # Redimensiona a imagem com base no nível de zoom
            zoomed_image = self.current_image_pil.resize(
                (int(self.current_image_pil.width * self.zoom_level),
                 int(self.current_image_pil.height * self.zoom_level)),
                Image.LANCZOS
            )
            self.tk_image = ImageTk.PhotoImage(zoomed_image)

            # Limpa o canvas e desenha a imagem com as coordenadas de pan
            self.canvas.delete("all")
            self.canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_image)
            self.canvas.image = self.tk_image  # Mantém referência da imagem

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

    def load_imagem_arquivo(self):
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem",
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp")],
        )
        if file_path:
            self.current_image_pil = Image.open(file_path)
            self.display_image(self.current_image_pil)
        self.paciente = None
        self.image_idx = None

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
            self.current_image_pil = load_image(self.images, self.paciente, self.image_idx)
            self.display_image(self.current_image_pil)
        except ValueError as e:
            print(e)

    def display_image(self, image):
        """Exibe a imagem no canvas e ajusta o tamanho do canvas para a imagem."""
        # Converte a imagem PIL para um objeto PhotoImage do Tkinter
        self.current_image_tk = ImageTk.PhotoImage(image)

        # Atualiza as dimensões do canvas para corresponder à imagem
        self.canvas.config(width=image.width, height=image.height)

        # Limpa o canvas antes de desenhar a nova imagem
        self.canvas.delete("all")

        # Exibe a imagem no canvas
        self.canvas.create_image(0, 0, anchor='nw', image=self.current_image_tk)

        # Mantém uma referência para evitar que a imagem seja coletada pelo garbage collector
        self.canvas.image = self.current_image_tk

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
        """Recortar a ROI e exibi-la, levando em consideração zoom e pan."""
        x1, y1 = self.roi_coords[0]
        x2, y2 = self.roi_coords[1]

        # Obtendo as dimensões do canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Obtendo o nível de zoom e a posição de pan
        zoom_level = self.zoom_level  # Supondo que você tenha um atributo que guarda o nível de zoom
        pan_x = self.pan_x  # Supondo que você tenha um atributo que guarda o deslocamento horizontal
        pan_y = self.pan_y  # Supondo que você tenha um atributo que guarda o deslocamento vertical

        # Calculando o fator de escala baseado no nível de zoom
        scale = zoom_level  # Ajuste esse valor se seu fator de zoom não for diretamente o zoom_level

        # Convertendo as coordenadas da ROI para a imagem original
        orig_x1 = (x1 - pan_x) / scale
        orig_y1 = (y1 - pan_y) / scale
        orig_x2 = (x2 - pan_x) / scale
        orig_y2 = (y2 - pan_y) / scale

        # Recortando a ROI da imagem original
        roi = crop_roi(
            self.current_image_pil, orig_x1, orig_y1, orig_x2, orig_y2, canvas_width, canvas_height
        )

        # Exibindo a ROI em uma nova janela
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
                show_features(features)

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

        output_path, hi,figado_normalizado = process_rois(roiRim, roiFigado, f"Rois/ROI_{self.paciente}_{self.image_idx}.png")

        # Determinar a classe (0-16 saudável, 17+ com esteatose)
        if self.paciente!=None:
            classe = 0 if self.paciente <= 16 else 1  # Pacientes 0 a 16 são saudáveis, restante com esteatose
        else:
            classe = input('Preencha a classe (0 para saudavel 1 para doente):')
        
        # Preparar os dados para escrita
        roi_data = [
            f"ROI_{self.paciente}_{self.image_idx}.png",
            classe,
            f"{x1_F},{y1_F}",  # Canto superior esquerdo do fígado
            f"{x1_R},{y1_R}",  # Canto superior esquerdo do rim
            hi
        ]

        self.save_kidney(roiFigado)
        self.save_liver(roiRim)
        
        # Salvar os dados no CSV
        write_roi_data_to_csv('Rois/Planilhas/roi_data.csv', roi_data)

                # Calcular características de textura para o fígado
        texture_features = calculate_texture_features(figado_normalizado)
        glcm_radial,features_glcm = compute_glcm(np.array(
                    figado_normalizado.convert("L")
                ))
        
        # Preparar os dados para escrita das características de textura
        texture_data = [
            f"ROI_{self.paciente}_{self.image_idx}.png",  # Nome do arquivo
            classe,
            texture_features['Coarseness'],
            texture_features['Contrast'],
            texture_features['Periodicity'],
            texture_features['Roughness'],
            
            features_glcm['contrast'][0][0], #[distancia = 1]
            features_glcm['dissimilarity'][0][0],
            features_glcm['homogeneity'][0][0],
            features_glcm['energy'][0][0],
            features_glcm['correlation'][0][0],
            features_glcm['ASM'][0][0],
            features_glcm['entropy'][0][0],
            
            features_glcm['contrast'][0][1], #[distancia = 2]
            features_glcm['dissimilarity'][0][1],
            features_glcm['homogeneity'][0][1],
            features_glcm['energy'][0][1],
            features_glcm['correlation'][0][1],
            features_glcm['ASM'][0][1],
            features_glcm['entropy'][0][1],
            
            features_glcm['contrast'][0][2], #[distancia = 4]
            features_glcm['dissimilarity'][0][2],
            features_glcm['homogeneity'][0][2],
            features_glcm['energy'][0][2],
            features_glcm['correlation'][0][2],
            features_glcm['ASM'][0][2],
            features_glcm['entropy'][0][2],
            
            features_glcm['contrast'][0][3], #[distancia = 8]
            features_glcm['dissimilarity'][0][3],
            features_glcm['homogeneity'][0][3],
            features_glcm['energy'][0][3],
            features_glcm['correlation'][0][3],
            features_glcm['ASM'][0][3],
            features_glcm['entropy'][0][3],
        ]

        # Escrever as características de textura no CSV
        write_texture_features_to_csv('Rois/Planilhas/texture_data.csv', texture_data)
    
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

    def save_kidney(self,image):
        try:
            pacient_int = int(self.paciente)  # Converte para inteiro para comparação
            image_int = int(self.image_idx)  # Converte para inteiro, se necessário

            # Cria o diretório se não existir
            save_directory = "Rois/Figado"
            os.makedirs(
                save_directory, exist_ok=True
            )  # Cria o diretório se não existir

            if self.paciente!=None:
                classe = 0 if self.paciente <= 16 else 1  # Pacientes 0 a 16 são saudáveis, restante com esteatose
            else:
                classe = input('Preencha a classe (0 para saudavel 1 para doente):')
            
            # Define o caminho do arquivo
            if self.paciente == None or self.image_idx == None:
                pacient_int = input("Escreva o numero do paciente :")
                image_int = input("Escreva qual é a imagem do paciente : ")
            
            
            # Define o caminho do arquivo
            if classe==0:
                file_path = os.path.join(
                    save_directory, f"SaudavelPacient{pacient_int}Imagem{image_int}.png"
                )
            else:
                file_path = os.path.join(
                    save_directory, f"DoentePacient{pacient_int}Imagem{image_int}.png"
                )

            # Salva a imagem
            new_size = (600, 600)  # Define o novo tamanho
            resized_image = image.resize(new_size)
            resized_image.save(file_path)
        except Exception as e:
            print(f"Erro ao salvar a imagem: {e}")

    def save_liver(self,image):
        try:
            print
            pacient_int = int(self.paciente)  # Converte para inteiro para comparação
            image_int = int(self.image_idx)  # Converte para inteiro, se necessário

            # Cria o diretório se não existir
            save_directory = "Rois/Rim"
            os.makedirs(
                save_directory, exist_ok=True
            )  # Cria o diretório se não existir

            # Define o caminho do arquivo
            if self.paciente == None or self.image_idx == None:
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
            resized_image = image.resize(new_size)
            resized_image.save(file_path)
        except Exception as e:
            print(f"Erro ao salvar a imagem: {e}")

# Iniciar a aplicação
root = tk.Tk()
app = ImageApp(root)
root.mainloop()
