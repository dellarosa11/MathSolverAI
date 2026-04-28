# Benchmarks

Use este diretorio para guardar imagens rotuladas e manifestos de benchmark.

## Manifesto minimo

Crie um arquivo como [samples.example.json](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/benchmarks/samples.example.json) e ajuste os caminhos para imagens reais:

```json
[
  {
    "image": "../data/raw/minha_equacao_01.png",
    "expected_expression": "1+2=3",
    "expected_result": "True"
  }
]
```

## Executando

```powershell
python src/utils/benchmark_inference.py --manifest benchmarks/samples.example.json --model models/math_cnn_plus_hasy_weights_best.pth --report-path benchmarks/report.json --csv-path benchmarks/report.csv
```

Observacao:

- o benchmark so faz sentido com imagens de expressoes reais, nao com folhas contendo simbolos soltos em varias linhas
- para diagnosticar uma imagem isolada antes de benchmark, prefira `main.py --diagnostic` ou `src/utils/export_inference_debug.py`
