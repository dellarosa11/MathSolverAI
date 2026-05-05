from __future__ import annotations

from typing import Dict, List


DIGIT_CLASSES: List[str] = [str(i) for i in range(10)]
DIGIT_FOLDER_TO_LABEL: Dict[str, str] = {label: label for label in DIGIT_CLASSES}

# Nomes canonicos usados no dataset em disco.
SYMBOL_FOLDER_TO_LABEL: Dict[str, str] = {
    "plus": "+",
    "minus": "-",
    "times": "*",
    "div": "/",
    "equals": "=",
    "lparen": "(",
    "rparen": ")",
}

FOLDER_TO_LABEL: Dict[str, str] = {
    **DIGIT_FOLDER_TO_LABEL,
    **SYMBOL_FOLDER_TO_LABEL,
}

SYMBOL_FOLDER_CLASSES: List[str] = list(SYMBOL_FOLDER_TO_LABEL.keys())
ALL_CLASSES: List[str] = DIGIT_CLASSES + [SYMBOL_FOLDER_TO_LABEL[name] for name in SYMBOL_FOLDER_CLASSES]
CLASS_TO_INDEX: Dict[str, int] = {label: idx for idx, label in enumerate(ALL_CLASSES)}
FOLDER_TO_INDEX: Dict[str, int] = {
    folder_name: CLASS_TO_INDEX[label]
    for folder_name, label in FOLDER_TO_LABEL.items()
}
LABEL_TO_FOLDER: Dict[str, str] = {label: label for label in DIGIT_CLASSES}
LABEL_TO_FOLDER.update({label: folder_name for folder_name, label in SYMBOL_FOLDER_TO_LABEL.items()})


def get_default_classes() -> List[str]:
    """Retorna a lista padrao de classes previstas pelo modelo."""
    return list(ALL_CLASSES)


def get_folder_name_for_label(label: str) -> str:
    """Converte um rotulo previsto no nome de pasta usado no dataset em disco."""
    if label not in LABEL_TO_FOLDER:
        raise KeyError(f"Rotulo nao suportado: {label}")
    return LABEL_TO_FOLDER[label]


def get_label_for_folder_name(folder_name: str) -> str:
    """Converte um nome de pasta do dataset em seu rotulo canonico."""
    if folder_name not in FOLDER_TO_LABEL:
        raise KeyError(f"Nome de pasta nao suportado: {folder_name}")
    return FOLDER_TO_LABEL[folder_name]
