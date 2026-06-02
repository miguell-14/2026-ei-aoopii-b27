# Garment Attribute Classification
Projeto desenvolvido no âmbito da unidade curricular de **AOOP II** (2026).

## Descrição
Este projeto tem como objetivo a classificação de atributos de peças de roupa a partir de imagens, utilizando técnicas de visão computacional e deep learning.

O sistema identifica em simultâneo os seguintes atributos:
- Comprimento da manga (sem mangas, manga curta, manga meia, manga comprida)
- Tipo de tecido (ganga, algodão, couro, pelo, malha, chiffon, outro)
- Padrão de cor (floral, estampado, riscas, cor sólida, xadrez, outro, blocos de cor)

O projeto aproxima-se de um problema de **classificação multi-output**, prevendo vários atributos em simultâneo a partir de uma única imagem.

## Objetivos
- Desenvolver um modelo de visão computacional para classificação de vestuário
- Aplicar técnicas de deep learning com Transfer Learning
- Avaliar e melhorar iterativamente o desempenho do modelo
- Disponibilizar uma demonstração interativa do modelo treinado

## Dataset
Este projeto utiliza o dataset **DeepFashion-MultiModal**, disponível no Kaggle.
O dataset contém imagens de pessoas vestidas com anotações manuais de atributos de vestuário.
Disponível em: [github.com/yumingj/DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal)

## Demonstração
A pasta `src/` contém uma aplicação Gradio que permite carregar uma imagem e obter as previsões do modelo em tempo real.

```bash
cd src
python app.py
```

## Ambiente de Desenvolvimento
Devido à dimensão do dataset, o desenvolvimento foi realizado utilizando **Kaggle Notebooks**, permitindo acesso direto aos dados sem necessidade de download local.

## Tecnologias
- Python
- PyTorch e Torchvision (EfficientNet-B0)
- Gradio
- Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- Kaggle Notebooks (GPU Tesla T4)

## Estado atual
| Notebook | Descrição | Estado |
|---|---|---|
| 01 — Exploration | Análise exploratória do dataset | Completo |
| 02 — Data Preparation | Limpeza, splits e pré-processamento | Completo |
| 03 — Model Training v1 | Primeira versão do treino do modelo | Completo |
| 04 — Evaluation | Avaliação do modelo no conjunto de teste | Completo |
| 05 — Conclusions | Análise de erros e melhorias propostas | Completo |
| 06 — Improved Training | Treino melhorado com focal loss | Completo |

## Autor
- **Nome:** Miguel Miranda Rebouço
- **Turma:** B
- **Email:** mir@ipvc.pt