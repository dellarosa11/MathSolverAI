# MathSolverAI

Projeto para reconhecer expressões matemáticas a partir de imagens, transformar o que foi escrito em texto e tentar resolver a expressão automaticamente.

## O que o projeto faz

- lê uma imagem com conta ou equação
- segmenta os símbolos
- reconhece números e operadores
- monta a expressão em texto
- tenta corrigir ambiguidades simples
- resolve a expressão com SymPy quando ela é válida

## O que foi usado

### Tecnologias

- Python
- PyTorch
- OpenCV
- SymPy
- NumPy
- scikit-learn
- Pillow
- Streamlit

### Modelos

- MLP nas versões iniciais
- CNN e CNN Plus nas versões mais recentes

### Datasets

- MNIST
- HASYv2
- BHMSDS
- MathWriting
- dados sintéticos gerados pelo próprio projeto
- imagens reais usadas para benchmark local

## Como instalar

```powershell
python -m pip install -r requirements.txt
```

## Como rodar

### Inferência padrão

```powershell
python main.py
```

### Inferência com parâmetros

```powershell
python main.py --image data/raw/teste.jpg --model models/math_mlp_weights.pth --diagnostic
```

### Interface web

```powershell
streamlit run app_streamlit.py
```

## Como treinar

```powershell
python src/train.py
```

## Como testar

```powershell
python -m pytest -q
```

## Estrutura principal

- `main.py`: executa o pipeline de reconhecimento e resolução
- `app_streamlit.py`: interface web para testar a IA
- `src/train.py`: treinamento dos modelos
- `src/preprocessor.py`: pré-processamento e segmentação
- `src/models/`: arquiteturas e carregamento de modelos
- `src/postprocessor.py`: correção da expressão reconhecida
- `src/solver.py`: resolução simbólica
- `benchmarks/`: arquivos de benchmark e comparação

## Status

O projeto já reconhece expressões simples, possui treino com CNN, benchmark local e interface para testes. O desempenho final depende bastante da qualidade da imagem, da segmentação e da variedade do dataset.

## Integrantes

- Guilherme Marolla Tesch
- Matheus Della Rosa
