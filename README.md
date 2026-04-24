# Garment Attribute Classification

Projeto desenvolvido no âmbito da unidade curricular de **AOOP II** (2026).

## Descrição
Este projeto tem como objetivo a classificação de atributos de peças de roupa a partir de imagens, utilizando técnicas de visão computacional e deep learning.

Numa fase inicial, o sistema irá identificar características como:

- Tipo de peça (ex: t-shirt, jeans, dress)
- Cor base (ex: blue, black, white)

O projeto poderá ser posteriormente expandido para incluir múltiplos atributos simultaneamente, aproximando-se de um problema de **classificação multi-label**.

## Objetivos
- Desenvolver um modelo de visão computacional para classificação de vestuário
- Aplicar técnicas de deep learning (ex: redes neuronais convolucionais)
- Explorar abordagens de Transfer Learning com modelos pré-treinados
- Avaliar o desempenho do modelo em dados reais

## Dataset
Este projeto utiliza o dataset **Fashion Product Images**, disponível no Kaggle.

O dataset contém:
- Imagens de produtos de moda
- Metadados associados (ex: tipo de peça, cor, género, utilização)

Atributos principais utilizados:
- `articleType`
- `baseColour`
- `usage`

## Ambiente de Desenvolvimento
Devido à dimensão do dataset, o desenvolvimento foi realizado utilizando **Kaggle Notebooks**, permitindo acesso direto aos dados sem necessidade de download local.

Os notebooks desenvolvidos no Kaggle serão posteriormente integrados neste repositório.

## Tecnologias (planeadas)
- Python
- PyTorch
- Modelos pré-treinados (ex: ResNet)
- Jupyter Notebook (Kaggle)


## Autor
- **Nome:** Miguel Miranda Rebouço  
- **Turma:** B  
- **Email:** mir@ipvc.pt