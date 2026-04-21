from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence

from torch.utils.data import ConcatDataset, Dataset
from torchvision import datasets, transforms

from src.data.class_config import FOLDER_TO_INDEX, SYMBOL_FOLDER_TO_LABEL, get_default_classes


def get_base_transform() -> transforms.Compose:
    """Transformacao usada de forma consistente no treino e na inferencia."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


class SymbolImageFolder(datasets.ImageFolder):
    """
    Dataset para simbolos customizados organizado em pastas por classe.
    """

    def __init__(self, root: str | Path, transform: Callable | None = None):
        super().__init__(root=str(root), transform=transform)

        invalid_classes = [name for name in self.classes if name not in FOLDER_TO_INDEX]
        if invalid_classes:
            allowed = ", ".join(get_default_classes())
            invalid = ", ".join(sorted(invalid_classes))
            raise ValueError(
                f"Classes nao reconhecidas no dataset customizado: {invalid}. "
                f"Classes permitidas: {allowed}."
            )

    def find_classes(self, directory: str):
        classes, class_to_idx = super().find_classes(directory)
        classes = sorted(classes, key=lambda name: FOLDER_TO_INDEX[name])
        class_to_idx = {name: FOLDER_TO_INDEX[name] for name in classes}
        return classes, class_to_idx


def _build_mnist_dataset(
    data_dir: Path,
    train: bool,
    transform: Callable | None,
) -> Dataset:
    return datasets.MNIST(
        root=str(data_dir),
        train=train,
        download=True,
        transform=transform,
    )


def _build_symbol_dataset(
    split_dir: Path,
    transform: Callable | None,
) -> Dataset | None:
    if not split_dir.exists():
        return None

    class_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
    if not class_dirs:
        return None

    return SymbolImageFolder(split_dir, transform=transform)


def build_math_dataset(
    data_dir: str | Path,
    train: bool = True,
    transform: Callable | None = None,
) -> Dataset:
    """
    Combina o MNIST com um dataset customizado de simbolos, se existir.
    """
    data_path = Path(data_dir)
    transform = transform or get_base_transform()

    datasets_to_concat: List[Dataset] = [
        _build_mnist_dataset(data_path, train=train, transform=transform)
    ]

    split_name = "train" if train else "val"
    symbol_split_dir = data_path / "symbols" / split_name
    symbol_dataset = _build_symbol_dataset(symbol_split_dir, transform=transform)
    if symbol_dataset is not None:
        datasets_to_concat.append(symbol_dataset)

    if len(datasets_to_concat) == 1:
        return datasets_to_concat[0]

    return ConcatDataset(datasets_to_concat)


def get_present_symbol_classes(data_dir: str | Path, split: str = "train") -> Sequence[str]:
    """Lista as classes de simbolos que ja existem em disco para o split informado."""
    split_dir = Path(data_dir) / "symbols" / split
    if not split_dir.exists():
        return []

    folder_names = [path.name for path in split_dir.iterdir() if path.is_dir()]
    return [
        SYMBOL_FOLDER_TO_LABEL[name]
        for name in sorted(folder_names, key=lambda name: FOLDER_TO_INDEX.get(name, 10_000))
        if name in SYMBOL_FOLDER_TO_LABEL
    ]
