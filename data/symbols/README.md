# Dataset de Simbolos Matematicos

Este diretorio guarda imagens customizadas de simbolos matematicos para complementar o `MNIST`.

## Estrutura esperada

```text
data/
  symbols/
    train/
      plus/
      minus/
      times/
      div/
      equals/
      lparen/
      rparen/
    val/
      plus/
      minus/
      times/
      div/
      equals/
      lparen/
      rparen/
```

## Mapeamento de pastas

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

## Observacao

Se `data/symbols/train` estiver vazio, o treino continua funcionando apenas com os digitos do `MNIST`.

## Aproveitando o dataset externo com `caption.txt`

Se voce tiver um dataset de expressoes completas rotuladas, pode extrair apenas expressoes simples e converter em simbolos isolados com:

```powershell
python src/data/prepare_symbols_from_captions.py --source-dir "C:\Users\mathe\Downloads\data" --clean-output
```

Ou, se o seu Windows usar o launcher:

```powershell
py src/data/prepare_symbols_from_captions.py --source-dir "C:\Users\mathe\Downloads\data" --clean-output
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
