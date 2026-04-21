# MathSolverAI - Reconhecimento de Equacoes Matematicas

## Descricao

Projeto em desenvolvimento com o objetivo de criar um sistema baseado em Inteligencia Artificial capaz de reconhecer simbolos e equacoes matematicas a partir de imagens.

## Objetivo

Desenvolver um modelo que seja capaz de:

- Identificar simbolos matematicos
- Interpretar equacoes simples
- Converter a equacao para formato digital (texto ou LaTeX)

## Dataset

Atualmente o projeto treina com:

- `MNIST` para os digitos `0-9`
- `data/symbols/train` e `data/symbols/val` para simbolos matematicos customizados

Os nomes de pasta esperados para simbolos estao em [data/symbols/README.md](C:\Users\mathe\Documents\GitHub\MathSolverAI\data\symbols\README.md).
Tambem existe um script para reaproveitar datasets com `caption.txt` em [src/data/prepare_symbols_from_captions.py](C:\Users\mathe\Documents\GitHub\MathSolverAI\src\data\prepare_symbols_from_captions.py).

## Tecnologias

- Python
- PyTorch ou TensorFlow
- OpenCV
- SymPy
- NumPy
- Pandas

## Configurando o Ambiente

1. Python instalado.
2. Clone o repositorio:
   `git clone https://github.com/dellarosa11/MathSolverAI.git`
3. Instale as dependencias:
   `pip install -r requirements.txt`

## Status

Em desenvolvimento

## Integrantes

- Guilherme Marolla Tesch RA: 113800
- Matheus Della Rosa RA: 113209

## Licenca

Este projeto e destinado exclusivamente a fins academicos e institucionais.
