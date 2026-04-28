# MathSolverAI

Projeto em desenvolvimento para reconhecer simbolos e expressoes matematicas a partir de imagens, montar a expressao em texto e tentar resolve-la com SymPy.

## Objetivo

- Identificar digitos e simbolos matematicos em imagens
- Reconhecer expressoes simples como `2+3` ou `x+2=5`
- Resolver a expressao reconhecida quando ela for valida
- Dar diagnostico suficiente para entender onde a IA esta acertando e onde esta confundindo classes

## Tecnologias

- Python
- PyTorch
- OpenCV
- SymPy
- NumPy
- scikit-learn
- Pillow

## Estrutura do Projeto

- [main.py](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/main.py): ponto de entrada para inferencia e diagnostico
- [src/train.py](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/src/train.py): treinamento com CLI, augmentation, sampler balanceado e relatorio JSON
- [src/preprocessor.py](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/src/preprocessor.py): pre-processamento, segmentacao e ordenacao dos simbolos
- [src/models/predictor.py](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/src/models/predictor.py): carregamento do modelo e predicao com confianca
- [src/postprocessor.py](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/src/postprocessor.py): corretor de expressao usando alternativas do top-k
- [src/solver.py](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/src/solver.py): validacao e resolucao simbolica com SymPy
- [src/data/generate_synthetic_symbols.py](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/src/data/generate_synthetic_symbols.py): gerador de numeros e operadores sinteticos
- [src/data/import_hasyv2.py](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/src/data/import_hasyv2.py): baixa o HASYv2 oficial e importa operadores manuscritos para `data/symbols`
- [src/utils/benchmark_inference.py](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/src/utils/benchmark_inference.py): benchmark com imagens rotuladas
- [src/utils/export_inference_debug.py](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/src/utils/export_inference_debug.py): exporta recortes, confianca e alternativas por simbolo
- [src/utils/import_debug_corrections.py](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/src/utils/import_debug_corrections.py): reaproveita recortes corrigidos como novos exemplos de treino
- [src/utils/analyze_training_report.py](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/src/utils/analyze_training_report.py): resumo de pior desempenho e maiores confusoes
- [data/symbols/README.md](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/data/symbols/README.md): nomes de pastas esperados para simbolos customizados

## Configuracao

Instale as dependencias:

```powershell
python -m pip install -r requirements.txt
```

## Como Testar

Rode os testes automatizados:

```powershell
python -m pytest -q
```

## Como Rodar a Inferencia

Usando os caminhos padrao:

```powershell
python main.py
```

Mostrando confianca por simbolo e alternativas:

```powershell
python main.py --image data/raw/teste.jpg --model models/math_mlp_weights.pth --diagnostic --top-k 3
```

Desligando o corretor de expressao para comparar o comportamento bruto do modelo:

```powershell
python main.py --image data/raw/teste.jpg --model models/math_mlp_weights.pth --disable-correction
```

Reconhecendo a expressao sem tentar resolver:

```powershell
python main.py --image data/raw/teste.jpg --model models/math_mlp_weights.pth --recognize-only
```

## Como Exportar os Erros da Inferencia

Para salvar os recortes de cada simbolo com confianca e alternativas:

```powershell
python src/utils/export_inference_debug.py --image data/raw/teste.jpg --model models/math_mlp_weights.pth --output-dir debug_outputs/minha_imagem
```

Esse comando gera:

- `annotated_predictions.png` com as caixas desenhadas
- um `summary.json` com a expressao reconhecida, a expressao corrigida e os candidatos do corretor
- um PNG do recorte cru e outro do recorte preparado para a rede em cada simbolo

## Como Reaproveitar Erros do Debug no Dataset

Se voce ja souber qual era a expressao correta da imagem debugada:

```powershell
python src/utils/import_debug_corrections.py --summary debug_outputs/minha_imagem/summary.json --expected-expression "1+2=3"
```

Se quiser importar apenas os simbolos que a IA errou:

```powershell
python src/utils/import_debug_corrections.py --summary debug_outputs/minha_imagem/summary.json --expected-expression "1+2=3" --mismatches-only
```

Se quiser aceitar a `corrected_expression` do summary como rotulo:

```powershell
python src/utils/import_debug_corrections.py --summary debug_outputs/minha_imagem/summary.json --use-corrected-expression
```

## Como Gerar Dados Sinteticos

Para gerar numeros e operadores automaticamente:

```powershell
python src/data/generate_synthetic_symbols.py --clean-output
```

Para gerar apenas operadores:

```powershell
python src/data/generate_synthetic_symbols.py --labels operators --train-count 300 --val-count 60
```

Para gerar um subconjunto:

```powershell
python src/data/generate_synthetic_symbols.py --labels 0 1 2 plus minus equals lparen rparen
```

## Como Importar o HASYv2 para Operadores Reais

O projeto tambem consegue baixar e integrar o [HASYv2](https://zenodo.org/records/259444), um dataset publico de simbolos manuscritos. No fluxo atual ele reforca especialmente:

- `plus` com `+`
- `minus` com `-`
- `times` com `\times`
- `div` com `/`

Importacao recomendada:

```powershell
python src/data/import_hasyv2.py
```

Se voce quiser reaproveitar o simbolo `\div` junto com `/` na classe de divisao:

```powershell
python src/data/import_hasyv2.py --include-obelus
```

Se quiser limpar apenas os arquivos importados anteriormente do HASY antes de rodar de novo:

```powershell
python src/data/import_hasyv2.py --clean-previous-import
```

Observacao importante:

- o HASYv2 ajuda bem em `+`, `-`, `*` e `/`
- ele nao cobre da forma que este projeto precisa as classes `=`, `(` e `)`
- por isso, continue usando seus dados sinteticos/manuais para essas classes

## Como Treinar

Treino recomendado com defaults mais fortes:

```powershell
python src/train.py
```

Esse fluxo agora usa por padrao:

- `cnn_plus`
- `learning-rate` menor
- augmentation no treino
- sampler balanceado
- label smoothing
- scheduler de learning rate
- early stopping
- relatorio JSON ao lado do checkpoint

Treino com parametros customizados:

```powershell
python src/train.py --epochs 20 --batch-size 32 --learning-rate 0.0005 --architecture cnn_plus --save-path models/math_cnn_weights.pth --report-path models/math_cnn_weights.json
```

Treino sem augmentation nem sampler balanceado:

```powershell
python src/train.py --disable-augmentation --disable-balanced-sampling
```

## Como Ler o Relatorio do Treino

Depois de treinar, voce pode resumir rapidamente o JSON:

```powershell
python src/utils/analyze_training_report.py models/math_cnn_weights.json
```

Isso mostra:

- melhor epoca
- melhor acuracia de validacao
- classes com pior desempenho
- maiores confusoes entre classes

## Como Medir a IA com Benchmark

Crie um manifesto `.json`, `.jsonl` ou `.csv` com campos `image` e `expected_expression`, e opcionalmente `expected_result`.

Exemplo em JSON:

```json
[
  {
    "image": "benchmarks/sample_01.png",
    "expected_expression": "1+2",
    "expected_result": "3"
  }
]
```

Depois rode:

```powershell
python src/utils/benchmark_inference.py --manifest benchmarks/samples.example.json --model models/math_cnn_weights.pth --report-path benchmarks/report.json --csv-path benchmarks/report.csv
```

O benchmark compara:

- expressao bruta reconhecida
- expressao corrigida pelo corretor top-k
- acerto exato da expressao
- acerto agregado por simbolo
- acerto do resultado quando `expected_result` existe
- quantas amostras melhoraram por causa do corretor

Veja tambem [benchmarks/README.md](/C:/Users/mathe/OneDrive/Documentos/GitHub/MathSolverAI/benchmarks/README.md) para um manifesto inicial.

## O Que Mais Ajuda a IA a Melhorar

- Aumentar a quantidade de simbolos em `data/symbols/train` e `data/symbols/val`
- Balancear melhor operadores como `+`, `-`, `=`, `(` e `)`
- Rodar treinos mais longos com `cnn`
- Usar o relatorio de confusao para descobrir quais classes estao se parecendo demais
- Gerar dados sinteticos para preencher lacunas de operadores e digitos
- Importar dados reais de operadores com o HASYv2 para sair do excesso de exemplos puramente sinteticos
- Exportar recortes de inferencia para enxergar se o problema esta na segmentacao ou na classificacao
- Corrigir recortes errados e importar isso de volta para `data/symbols/train`
- Rodar benchmark antes e depois de cada treino para medir ganho real

## Artefatos Locais

Arquivos como checkpoints `.pth`, caches de Python, imagens locais e dados baixados do MNIST ou do HASYv2 sao tratados como artefatos locais e nao devem ser versionados no git.

## Status Atual

- Inferencia via linha de comando pronta
- Modo diagnostico com confianca por simbolo pronto
- Corretor de expressao usando top-k pronto
- Exportacao de recortes para inspecao pronta
- Importacao de correcoes do debug para o dataset pronta
- Treinamento com augmentation, sampler balanceado e early stopping pronto
- Scheduler de learning rate e label smoothing prontos
- Geracao sintetica de numeros e operadores pronta
- Benchmark com imagens rotuladas pronto
- Relatorio com distribuicao por classe e maiores confusoes pronto
- Testes automatizados basicos prontos
- A qualidade final ainda depende muito de ter mais simbolos reais no dataset

## Integrantes

- Guilherme Marolla Tesch RA: 113800
- Matheus Della Rosa RA: 113209

## Licenca

Projeto destinado a fins academicos e institucionais.
