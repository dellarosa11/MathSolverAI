from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data.class_config import get_default_classes
from src.data.dataset_builder import (
    build_math_dataset,
    build_weighted_sampler,
    get_class_distribution,
    get_present_symbol_classes,
)
from src.models.model_factory import DEFAULT_ARCHITECTURE, build_model


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ModelTrainer:
    """
    Classe responsavel pelo treinamento e avaliacao do classificador de simbolos.
    """

    def __init__(
        self,
        epochs: int = 5,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        architecture: str = DEFAULT_ARCHITECTURE,
        device: Optional[str] = None,
        num_workers: int = 0,
        seed: int = 42,
        use_augmentation: bool = True,
        balanced_sampling: bool = True,
        early_stopping_patience: int = 5,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
        lr_scheduler_patience: int = 2,
        lr_scheduler_factor: float = 0.5,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.num_workers = num_workers
        self.seed = seed
        self.use_augmentation = use_augmentation
        self.balanced_sampling = balanced_sampling
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = get_default_classes()

        set_global_seed(self.seed)

        self.model = build_model(
            architecture=self.architecture,
            num_classes=len(self.class_names),
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
        )

    def _create_checkpoint(self, history: list[dict[str, Any]], best_val_accuracy: float) -> dict[str, Any]:
        checkpoint: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
            "architecture": self.architecture,
            "history": history,
            "best_val_accuracy": best_val_accuracy,
            "training_config": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "num_workers": self.num_workers,
                "seed": self.seed,
                "use_augmentation": self.use_augmentation,
                "balanced_sampling": self.balanced_sampling,
                "early_stopping_patience": self.early_stopping_patience,
                "weight_decay": self.weight_decay,
                "label_smoothing": self.label_smoothing,
                "lr_scheduler_patience": self.lr_scheduler_patience,
                "lr_scheduler_factor": self.lr_scheduler_factor,
            },
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if hasattr(self.model, "hidden_size"):
            checkpoint["hidden_size"] = self.model.hidden_size
        return checkpoint

    def _build_datasets(self, data_dir: str | Path) -> tuple[Dataset, Dataset]:
        train_dataset = build_math_dataset(
            data_dir=data_dir,
            train=True,
            use_augmentation=self.use_augmentation,
        )
        val_dataset = build_math_dataset(
            data_dir=data_dir,
            train=False,
            use_augmentation=False,
        )
        return train_dataset, val_dataset

    def _build_dataloader(self, dataset: Dataset, train: bool) -> DataLoader:
        sampler = None
        shuffle = train

        if train and self.balanced_sampling:
            sampler = build_weighted_sampler(dataset)
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.device.type == "cuda",
        )

    def _build_top_confusions(self, confusion: list[list[int]], limit: int = 10) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        for true_index, row in enumerate(confusion):
            for predicted_index, count in enumerate(row):
                if true_index == predicted_index or count <= 0:
                    continue
                pairs.append(
                    {
                        "true_label": self.class_names[true_index],
                        "predicted_label": self.class_names[predicted_index],
                        "count": int(count),
                    }
                )

        pairs.sort(key=lambda item: item["count"], reverse=True)
        return pairs[:limit]

    def _evaluate(self, dataloader: DataLoader) -> dict[str, Any]:
        self.model.eval()

        total_loss = 0.0
        total_examples = 0
        correct = 0
        all_labels: list[int] = []
        all_predictions: list[int] = []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                predictions = outputs.argmax(dim=1)
                batch_size = labels.size(0)

                total_loss += loss.item() * batch_size
                total_examples += batch_size
                correct += (predictions == labels).sum().item()

                all_labels.extend(labels.cpu().tolist())
                all_predictions.extend(predictions.cpu().tolist())

        avg_loss = total_loss / max(total_examples, 1)
        accuracy = correct / max(total_examples, 1)

        matrix = confusion_matrix(
            all_labels,
            all_predictions,
            labels=list(range(len(self.class_names))),
        )
        matrix_list = matrix.tolist()

        per_class_accuracy: dict[str, float | None] = {}
        for index, class_name in enumerate(self.class_names):
            class_total = int(matrix[index].sum())
            if class_total == 0:
                per_class_accuracy[class_name] = None
            else:
                per_class_accuracy[class_name] = float(matrix[index, index] / class_total)

        self.model.train()
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "confusion_matrix": matrix_list,
            "per_class_accuracy": per_class_accuracy,
            "most_confused_pairs": self._build_top_confusions(matrix_list),
        }

    def _print_worst_classes(self, per_class_accuracy: dict[str, float | None], limit: int = 5) -> None:
        ranked = [
            (class_name, accuracy)
            for class_name, accuracy in per_class_accuracy.items()
            if accuracy is not None
        ]
        ranked.sort(key=lambda item: item[1])

        print("[INFO] Classes com pior desempenho na validacao:")
        for class_name, accuracy in ranked[:limit]:
            print(f"  {class_name}: {accuracy * 100:.2f}%")

    def _print_top_confusions(self, confused_pairs: list[dict[str, Any]], limit: int = 5) -> None:
        if not confused_pairs:
            print("[INFO] Nenhuma confusao relevante encontrada na validacao.")
            return

        print("[INFO] Maiores confusoes observadas na validacao:")
        for pair in confused_pairs[:limit]:
            print(
                f"  {pair['true_label']} -> {pair['predicted_label']}: "
                f"{pair['count']} ocorrencias"
            )

    def _write_training_report(
        self,
        report_path: Path,
        history: list[dict[str, Any]],
        best_val_accuracy: float,
        best_epoch: int,
        final_val_metrics: dict[str, Any],
        train_class_distribution: dict[str, int],
        val_class_distribution: dict[str, int],
        stopped_early: bool,
    ) -> None:
        report = {
            "architecture": self.architecture,
            "device": str(self.device),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "use_augmentation": self.use_augmentation,
            "balanced_sampling": self.balanced_sampling,
            "early_stopping_patience": self.early_stopping_patience,
            "weight_decay": self.weight_decay,
            "label_smoothing": self.label_smoothing,
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "class_names": self.class_names,
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "history": history,
            "train_class_distribution": train_class_distribution,
            "val_class_distribution": val_class_distribution,
            "final_validation": final_val_metrics,
        }

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[INFO] Relatorio de treino salvo em: {report_path}")

    def train(self, data_dir: str | Path, save_path: str | Path, report_path: str | Path | None = None) -> dict[str, Any]:
        data_dir = Path(data_dir)
        save_path = Path(save_path)
        report_file = Path(report_path) if report_path is not None else save_path.with_suffix(".json")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        best_save_path = save_path.with_name(f"{save_path.stem}_best{save_path.suffix}")

        symbol_classes = get_present_symbol_classes(data_dir, split="train")
        val_symbol_classes = get_present_symbol_classes(data_dir, split="val")
        train_dataset, val_dataset = self._build_datasets(data_dir)
        train_class_distribution = get_class_distribution(train_dataset, self.class_names)
        val_class_distribution = get_class_distribution(val_dataset, self.class_names)
        train_loader = self._build_dataloader(train_dataset, train=True)
        val_loader = self._build_dataloader(val_dataset, train=False)

        print(f"[INFO] Classes configuradas: {', '.join(self.class_names)}")
        print(f"[INFO] Simbolos customizados encontrados no treino: {', '.join(symbol_classes) if symbol_classes else 'nenhum'}")
        print(f"[INFO] Simbolos customizados encontrados na validacao: {', '.join(val_symbol_classes) if val_symbol_classes else 'nenhum'}")
        print(f"[INFO] Arquitetura em uso: {self.architecture.upper()}")
        print(f"[INFO] Dataset raiz: {data_dir}")
        print(f"[INFO] Augmentation de treino: {'ligada' if self.use_augmentation else 'desligada'}")
        print(f"[INFO] Sampler balanceado: {'ligado' if self.balanced_sampling else 'desligado'}")
        print(f"[INFO] Iniciando treinamento em: {self.device}")

        self.model.train()

        history: list[dict[str, Any]] = []
        best_val_accuracy = -1.0
        best_epoch = 0
        epochs_without_improvement = 0
        stopped_early = False

        for epoch in range(self.epochs):
            running_loss = 0.0
            total_examples = 0
            correct = 0
            loop = tqdm(train_loader, leave=True)

            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                predictions = outputs.argmax(dim=1)
                batch_size = labels.size(0)

                running_loss += loss.item() * batch_size
                total_examples += batch_size
                correct += (predictions == labels).sum().item()

                loop.set_description(f"Epoch [{epoch + 1}/{self.epochs}]")
                loop.set_postfix(loss=loss.item())

            train_loss = running_loss / max(total_examples, 1)
            train_accuracy = correct / max(total_examples, 1)
            val_metrics = self._evaluate(val_loader)

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            history.append(epoch_record)

            print(
                f"[INFO] Epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_accuracy * 100:.2f}% "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy'] * 100:.2f}% "
                f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
            )

            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(self._create_checkpoint(history, best_val_accuracy), best_save_path)
                print(f"[INFO] Novo melhor checkpoint salvo em: {best_save_path}")
            else:
                epochs_without_improvement += 1

            if self.scheduler is not None:
                self.scheduler.step(val_metrics["loss"])

            if self.early_stopping_patience > 0 and epochs_without_improvement >= self.early_stopping_patience:
                stopped_early = True
                print(
                    "[INFO] Early stopping ativado: "
                    f"{epochs_without_improvement} epocas sem melhora na validacao."
                )
                break

        final_checkpoint = self._create_checkpoint(history, best_val_accuracy)
        torch.save(final_checkpoint, save_path)
        print(f"\n[SUCESSO] Treino concluido! Checkpoint final salvo em: {save_path}")
        print(f"[SUCESSO] Melhor checkpoint salvo em: {best_save_path}")

        final_val_metrics = self._evaluate(val_loader)
        print(f"[INFO] Melhor acuracia de validacao observada: {best_val_accuracy * 100:.2f}%")
        self._print_worst_classes(final_val_metrics["per_class_accuracy"])
        self._print_top_confusions(final_val_metrics["most_confused_pairs"])
        self._write_training_report(
            report_file,
            history,
            best_val_accuracy,
            best_epoch,
            final_val_metrics,
            train_class_distribution,
            val_class_distribution,
            stopped_early,
        )

        return {
            "history": history,
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "report_path": report_file,
            "save_path": save_path,
            "best_save_path": best_save_path,
            "stopped_early": stopped_early,
        }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina o classificador de simbolos do MathSolverAI.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(BASE_DIR / "data"),
        help="Diretorio raiz com o MNIST e os simbolos customizados.",
    )
    parser.add_argument(
        "--save-path",
        default=str(BASE_DIR / "models" / "math_mlp_weights.pth"),
        help="Caminho do checkpoint final a ser salvo.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Caminho opcional do relatorio JSON. Se vazio, usa o mesmo nome do checkpoint com extensao .json.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Numero de epocas.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tamanho do batch.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
        help="Learning rate do otimizador.",
    )
    parser.add_argument(
        "--architecture",
        choices=("cnn", "cnn_plus", "cnn_bn", "enhanced_cnn", "mlp"),
        default=DEFAULT_ARCHITECTURE,
        help="Arquitetura do modelo.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Device forcado, por exemplo 'cpu' ou 'cuda'. Se vazio, escolhe automaticamente.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Numero de workers do DataLoader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reproducibilidade.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay usado no AdamW.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing usado na CrossEntropyLoss.",
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=2,
        help="Paciencia do scheduler de learning rate baseado na validacao.",
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=0.5,
        help="Fator de reducao do learning rate no scheduler.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Numero de epocas sem melhora antes de interromper. Use 0 para desativar.",
    )
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="Desliga augmentation no conjunto de treino.",
    )
    parser.add_argument(
        "--disable-balanced-sampling",
        action="store_true",
        help="Desliga o sampler balanceado de classes no treino.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    trainer = ModelTrainer(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        architecture=args.architecture,
        device=args.device or None,
        num_workers=args.num_workers,
        seed=args.seed,
        use_augmentation=not args.disable_augmentation,
        balanced_sampling=not args.disable_balanced_sampling,
        early_stopping_patience=args.early_stopping_patience,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
    )
    trainer.train(
        data_dir=args.data_dir,
        save_path=args.save_path,
        report_path=args.report_path or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
