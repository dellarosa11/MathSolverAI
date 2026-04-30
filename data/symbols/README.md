# Dataset de Simbolos Matematicos

Este diretorio guarda imagens customizadas para complementar o `MNIST`.

## Estrutura esperada

Voce pode manter apenas operadores ou tambem incluir digitos sinteticos/customizados.

```text
data/
  symbols/
    train/
      0/
      1/
      ...
      9/
      plus/
      minus/
      times/
      div/
      equals/
      lparen/
      rparen/
    val/
      0/
      1/
      ...
      9/
      plus/
      minus/
      times/
      div/
      equals/
      lparen/
      rparen/
```

## Mapeamento de pastas

- `0` -> `0`
- `1` -> `1`
- `2` -> `2`
- `3` -> `3`
- `4` -> `4`
- `5` -> `5`
- `6` -> `6`
- `7` -> `7`
- `8` -> `8`
- `9` -> `9`
- `plus` -> `+`
- `minus` -> `-`
- `times` -> `*`
- `div` -> `/`
- `equals` -> `=`
- `lparen` -> `(`
- `rparen` -> `)`

## Regras para as imagens

- Formatos aceitos: `.png`, `.jpg`, `.jpeg`, `.bmp`
- Preferencia por fundo escuro e simbolo claro, semelhante ao preprocessamento atual
- O pipeline redimensiona automaticamente para `28x28`
- Quanto mais variacao de escrita, espessura e inclinacao, melhor

## Geracao sintetica automatica

Para gerar numeros e operadores automaticamente:

```powershell
python src/data/generate_synthetic_symbols.py --clean-output
```

Para gerar apenas operadores:

```powershell
python src/data/generate_synthetic_symbols.py --labels operators --train-count 300 --val-count 60
```

Para gerar apenas algumas classes:

```powershell
python src/data/generate_synthetic_symbols.py --labels 0 1 2 plus minus equals lparen rparen
```

O script:

- descobre fontes do sistema
- renderiza variacoes sinteticas com jitter, rotacao e ruido leve
- salva exemplos em `data/symbols/train` e `data/symbols/val`
- grava um manifesto em `data/symbols/synthetic_manifest.json`

## Importando operadores reais do HASYv2

Para baixar e importar operadores manuscritos do HASYv2:

```powershell
python src/data/import_hasyv2.py
```

O importador usa o arquivo oficial do Zenodo e, por padrao, reforca:

- `plus` com exemplos de `+`
- `minus` com exemplos de `-`
- `times` com exemplos de `\times`
- `div` com exemplos de `/`

Se quiser tambem juntar o simbolo `\div` na classe `div`:

```powershell
python src/data/import_hasyv2.py --include-obelus
```

Se quiser rodar novamente limpando apenas os arquivos `hasyv2_*.png` gerados antes:

```powershell
python src/data/import_hasyv2.py --clean-previous-import
```

Observacoes:

- o HASYv2 e otimo para reforcar alguns operadores reais
- ele nao resolve sozinho `equals`, `lparen` e `rparen`
- continue misturando HASYv2 com dados sinteticos, recortes corrigidos e exemplos manuais

## Importando BHMSDS para digitos e sinais

Para baixar e importar o BHMSDS:

```powershell
python src/data/import_bhmsds.py --clean-previous-import
```

Esse importador reforca:

- `0` a `9`
- `plus` com exemplos reais de `+`
- `minus` com exemplos reais de `-`
- `div` com exemplos reais de `/`

As imagens do BHMSDS sao invertidas automaticamente para combinar com o padrao do pipeline atual.

Para importar apenas operadores:

```powershell
python src/data/import_bhmsds.py --symbols operators --clean-previous-import
```

Para reforcar apenas `0`, `+`, `-` e `/`:

```powershell
python src/data/import_bhmsds.py --symbols 0 plus minus div --clean-previous-import
```

Observacoes:

- o BHMSDS nao cobre `equals`, `lparen` e `rparen`
- o dataset tambem nao oferece um simbolo `times` compativel com a classe `*` do projeto
- use BHMSDS como reforco para digitos e alguns sinais, nao como substituto unico do restante do dataset

## Reaproveitando erros do debug como treino

Depois de rodar o exportador de debug:

```powershell
python src/utils/export_inference_debug.py --image data/raw/teste.jpg --model models/math_mlp_weights.pth --output-dir debug_outputs/teste_01
```

Voce pode transformar os recortes corrigidos em novos exemplos do dataset:

```powershell
python src/utils/import_debug_corrections.py --summary debug_outputs/teste_01/summary.json --expected-expression "1+2=3" --mismatches-only
```

Isso copia os recortes preparados para a rede para as pastas corretas em `data/symbols/train`.

## Aproveitando o dataset externo com `caption.txt`

Se voce tiver um dataset de expressoes completas rotuladas, pode extrair apenas expressoes simples e converter em simbolos isolados com:

```powershell
python src/data/prepare_symbols_from_captions.py --source-dir "C:\Users\mathe\Downloads\data" --clean-output
```

O script:

- le todos os `caption.txt`
- mantem apenas expressoes simples compostas por digitos e `+ - * / = ( )`
- segmenta a imagem
- salva os simbolos extraidos em `data/symbols/train` e `data/symbols/val`

Parametros uteis:

- `--max-samples 500` para testar com poucas expressoes
- `--val-ratio 0.2` para definir a proporcao de validacao
- `--max-tokens 9` para limitar o tamanho das expressoes aceitas
