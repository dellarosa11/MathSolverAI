from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, List, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms

from src.data.class_config import FOLDER_TO_INDEX, FOLDER_TO_LABEL, get_default_classes


def get_base_transform(train: bool = False, use_augmentation: bool = False) -> transforms.Compose:
    """Transformacao usada de forma consistente no treino e na inferencia."""
    transform_steps: list[Callable] = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
    ]

    if train and use_augmentation:
        transform_steps.extend(
            [
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.08, 0.08),
                    scale=(0.92, 1.08),
                    shear=6,
                    fill=0,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                    p=0.15,
                ),
            ]
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return transforms.Compose(transform_steps)


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
    use_augmentation: bool = False,
) -> Dataset:
    """
    Combina o MNIST com um dataset customizado de simbolos, se existir.
    """
    data_path = Path(data_dir)
    transform = transform or get_base_transform(train=train, use_augmentation=use_augmentation)

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


def get_dataset_targets(dataset: Dataset) -> list[int]:
    """
    Extrai os rotulos inteiros de datasets nativos do torchvision e ConcatDataset.
    """
    if isinstance(dataset, ConcatDataset):
        targets: list[int] = []
        for subdataset in dataset.datasets:
            targets.extend(get_dataset_targets(subdataset))
        return targets

    if hasattr(dataset, "targets"):
        raw_targets = dataset.targets
        if hasattr(raw_targets, "tolist"):
            return [int(label) for label in raw_targets.tolist()]
        return [int(label) for label in raw_targets]

    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]

    raise TypeError(f"Dataset nao suportado para extracao de targets: {type(dataset)!r}")


def get_class_distribution(dataset: Dataset, class_names: Sequence[str]) -> dict[str, int]:
    """
    Conta quantos exemplos existem por classe no dataset combinado.
    """
    counts = Counter(get_dataset_targets(dataset))
    return {
        class_name: int(counts.get(class_index, 0))
        for class_index, class_name in enumerate(class_names)
    }


def build_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """
    Cria um sampler balanceado por frequencia de classe.
    """
    targets = get_dataset_targets(dataset)
    class_counts = Counter(targets)

    if not class_counts:
        raise ValueError("Nao foi possivel construir o sampler balanceado sem classes no dataset.")

    sample_weights = torch.DoubleTensor(
        [1.0 / class_counts[int(target)] for target in targets]
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_present_symbol_classes(data_dir: str | Path, split: str = "train") -> Sequence[str]:
    """Lista as classes customizadas que ja existem em disco para o split informado."""
    split_dir = Path(data_dir) / "symbols" / split
    if not split_dir.exists():
        return []

    folder_names = [path.name for path in split_dir.iterdir() if path.is_dir()]
    return [
        FOLDER_TO_LABEL[name]
        for name in sorted(folder_names, key=lambda name: FOLDER_TO_INDEX.get(name, 10_000))
        if name in FOLDER_TO_LABEL
    ]
