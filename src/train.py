from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data.class_config import get_default_classes
from src.data.dataset_builder import build_math_dataset, get_present_symbol_classes
from src.models.model_factory import DEFAULT_ARCHITECTURE, build_model


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
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = get_default_classes()

        self.model = build_model(
            architecture=self.architecture,
            num_classes=len(self.class_names),
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _get_dataloader(self, data_dir: str | Path, train: bool) -> DataLoader:
        """
        Prepara e retorna o DataLoader do dataset combinado.
        """
        dataset = build_math_dataset(data_dir, train=train)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train)

    def _create_checkpoint(self, history: list[dict], best_val_accuracy: float) -> dict:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
            "architecture": self.architecture,
            "history": history,
            "best_val_accuracy": best_val_accuracy,
        }
        if hasattr(self.model, "hidden_size"):
            checkpoint["hidden_size"] = self.model.hidden_size
        return checkpoint

    def _evaluate(self, dataloader: DataLoader) -> dict:
        """
        Avalia o modelo no conjunto fornecido e retorna metricas consolidadas.
        """
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

        per_class_accuracy: dict[str, float | None] = {}
        for idx, class_name in enumerate(self.class_names):
            class_total = int(matrix[idx].sum())
            if class_total == 0:
                per_class_accuracy[class_name] = None
            else:
                per_class_accuracy[class_name] = float(matrix[idx, idx] / class_total)

        self.model.train()
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "confusion_matrix": matrix,
            "per_class_accuracy": per_class_accuracy,
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

    def train(self, data_dir: str | Path, save_path: str | Path) -> None:
        """
        Executa treino, validacao por epoca e salva o melhor checkpoint.
        """
        data_dir = Path(data_dir)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        best_save_path = save_path.with_name(f"{save_path.stem}_best{save_path.suffix}")

        symbol_classes = get_present_symbol_classes(data_dir, split="train")
        val_symbol_classes = get_present_symbol_classes(data_dir, split="val")
        print(f"[INFO] Classes configuradas: {', '.join(self.class_names)}")
        print(f"[INFO] Simbolos customizados encontrados no treino: {', '.join(symbol_classes) if symbol_classes else 'nenhum'}")
        print(f"[INFO] Simbolos customizados encontrados na validacao: {', '.join(val_symbol_classes) if val_symbol_classes else 'nenhum'}")
        print(f"[INFO] Arquitetura em uso: {self.architecture.upper()}")

        train_loader = self._get_dataloader(data_dir, train=True)
        val_loader = self._get_dataloader(data_dir, train=False)

        print(f"[INFO] Iniciando treinamento em: {self.device}")
        self.model.train()

        history: list[dict] = []
        best_val_accuracy = -1.0

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
            }
            history.append(epoch_record)

            print(
                f"[INFO] Epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_accuracy * 100:.2f}% "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy'] * 100:.2f}%"
            )

            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                torch.save(self._create_checkpoint(history, best_val_accuracy), best_save_path)
                print(f"[INFO] Novo melhor checkpoint salvo em: {best_save_path}")

        final_checkpoint = self._create_checkpoint(history, best_val_accuracy)
        torch.save(final_checkpoint, save_path)
        print(f"\n[SUCESSO] Treino concluido! Checkpoint final salvo em: {save_path}")
        print(f"[SUCESSO] Melhor checkpoint salvo em: {best_save_path}")

        final_val_metrics = self._evaluate(val_loader)
        print(f"[INFO] Melhor acuracia de validacao observada: {best_val_accuracy * 100:.2f}%")
        self._print_worst_classes(final_val_metrics["per_class_accuracy"])


def main():
    """
    Ponto de entrada para execucao do treinamento.
    """
    data_dir = BASE_DIR / "data"
    save_path = BASE_DIR / "models" / "math_mlp_weights.pth"

    trainer = ModelTrainer(epochs=5, batch_size=64, learning_rate=0.001)
    trainer.train(data_dir, save_path)


if __name__ == "__main__":
    main()
