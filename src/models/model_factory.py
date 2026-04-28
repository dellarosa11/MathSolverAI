from __future__ import annotations

from src.models.cnn_model import MathCNN, MathCNNPlus
from src.models.mlp_model import MathMLP


DEFAULT_ARCHITECTURE = "cnn_plus"


def build_model(architecture: str, num_classes: int, hidden_size: int = 128):
    """
    Constroi a arquitetura solicitada com parametros compativeis com checkpoint.
    """
    normalized = architecture.lower()
    if normalized == "cnn":
        return MathCNN(num_classes=num_classes)
    if normalized in {"cnn_plus", "cnn_bn", "enhanced_cnn"}:
        return MathCNNPlus(num_classes=num_classes)
    if normalized == "mlp":
        return MathMLP(hidden_size=hidden_size, num_classes=num_classes)
    raise ValueError(f"Arquitetura nao suportada: {architecture}")
