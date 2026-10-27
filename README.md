# Diagnóstico de esteatose hepática em ultrassom

Projeto acadêmico para explorar imagens B-mode do fígado. Inclui uma interface desktop para selecionar regiões de interesse (ROIs), normalização pelo índice hepatorrenal, extração de características de textura e experimentos de classificação com MobileNet e SVM.

> Projeto de pesquisa/estudo. Não é dispositivo médico e não deve orientar diagnóstico ou tratamento.

## Estrutura

| Caminho | Conteúdo |
| --- | --- |
| `main.py`, `RoiApp.py` | Interface Tkinter para abrir imagens e selecionar ROIs. |
| `actions/` | Leitura do conjunto `.mat`, recorte, normalização, características, gráficos e gravação dos CSVs. |
| `data/dataset_liver_bmodes_steatosis_assessment_IJCARS.mat` | Conjunto de imagens usado pela interface. |
| `Artigo.pdf` | Relatório do projeto, metodologia e resultados que motivaram os experimentos. |
| `Rois/Normalizadas/` | Imagens usadas pelos experimentos CNN. |
| `roi_data.csv` | Índices e rótulos usados pelos scripts MobileNet legados. |
| `Rois/Planilhas/texture_data.csv` | Características de textura usadas pelo SVM. |
| `MobileNetUp.py` | Experimento MobileNet principal preservado no projeto. |
| `a.py` | Experimento antigo e duplicado; não é o ponto de entrada recomendado. |
| `MobileNetPeso10.py` | Variante experimental MobileNet com pesos de classe definidos no código. |
| `melhor_modelo_balanceamento_fixo/` | Variante de treino com balanceamento fixo e seus resultados salvos. |
| `melhor_modelo_balanceamento_dinamico/` | Avaliação de transformações de imagem e resultados do experimento dinâmico. |
| `svm.py` | SVM sobre características de textura, com validação por paciente. |
| `teste2.py`, `teste3.py` | Protótipos antigos de predição/avaliação. Ambos esperam `MobileNet.h5` na raiz, arquivo que não existe ali. |
| `*.h5`, `MobileNet_SavedModel/` | Pesos/modelos treinados preservados no repositório. |
| `best_model_details.txt`, `results.txt`, `melhores.txt`, `confusion_matrices.txt` | Registros de experimentos anteriores. |

Os scripts e artefatos são experimentais; nem todos formam um único fluxo reproduzível. Vários caminhos são relativos à raiz do projeto, então execute os comandos abaixo a partir dela.

## Ambiente

Use Python 3.11. No Windows, crie um ambiente virtual e instale as dependências:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Tkinter costuma vir com o instalador oficial do Python para Windows. Em Linux, instale o pacote Tk do sistema se a interface não abrir.

## Executar

Abra a interface de seleção de imagens e ROIs:

```bash
python main.py
```

Execute os experimentos separadamente, sempre a partir da raiz:

```bash
python svm.py
python MobileNetUp.py
python MobileNetPeso10.py
python -m melhor_modelo_balanceamento_fixo.MobileNetUp
```

Os scripts MobileNet podem levar bastante tempo e consumir memória. Alguns carregam MobileNet com pesos ImageNet; a primeira execução pode precisar baixar esses pesos. Os scripts de avaliação em `melhor_modelo_balanceamento_dinamico/`, `teste2.py` e `teste3.py` também dependem dos caminhos e modelos definidos no próprio arquivo.

Rode as verificações leves com:

```bash
python -m unittest discover -s tests -v
```

## Dados e resultados

Ao salvar uma ROI com o mesmo nome, os CSVs agora atualizam a linha correspondente em vez de criar uma amostra duplicada. O treino CNN separa pacientes de treino, validação e teste; o SVM mantém pacientes agrupados na validação interna e ajusta a escala dentro de cada fold.

### Artigo e arquivos antigos

O artigo descreve 55 pacientes e 550 imagens, com 10 imagens por paciente. O conjunto original está identificado no [Zenodo](https://doi.org/10.5281/zenodo.1009146); o registro lista o arquivo, mas não mostra uma licença de redistribuição.

Os resultados do artigo precisam de revisão antes de serem reutilizados: a matriz do SVM `[TN=1, FP=169, FN=1, TP=379]` implica sensibilidade de 99,74%, especificidade de 0,59%, precisão de 69,16% e F1 de 81,68%, diferentes dos valores no texto. A matriz da MobileNet soma 560 casos, embora o artigo descreva 550 imagens. O código atual também usa validação agrupada por paciente, então seus resultados precisam ser recalculados; eles não reproduzem automaticamente os números do artigo. `a.py` é uma tentativa antiga de treino e `teste2.py`/`teste3.py` são protótipos ligados a um nome de modelo ausente. Mantive esses arquivos para referência, sem recomendá-los como fluxo principal.

O projeto contém imagens médicas, CSVs derivados e modelos binários. Antes de publicar um repositório público, obtenha confirmação dos autores sobre redistribuição do conjunto e derivados. A licença do código deste repositório também ainda não foi definida. Os binários já rastreados continuam no histórico; `.gitignore` não os remove nem apaga versões anteriores.

O README antigo também apontava para uma pasta externa com ROIs e planilhas: [Google Drive](https://drive.google.com/drive/folders/17BSZ7plBQsZMAGguSzub70X_KXOZr3Fd).
