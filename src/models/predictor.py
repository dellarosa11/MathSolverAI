from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

# Adiciona a raiz do projeto (duas pastas acima de predictor.py) ao path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data.class_config import get_default_classes
from src.data.dataset_builder import get_base_transform
from src.models.model_factory import build_model


class MathPredictor:
    """
    Classe responsavel por realizar a inferencia.
    """

    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            class_names = checkpoint.get("class_names", get_default_classes())
            hidden_size = checkpoint.get("hidden_size", 128)
            architecture = checkpoint.get("architecture", "mlp")
            state_dict = checkpoint["model_state_dict"]
        else:
            class_names = get_default_classes()[:10]
            hidden_size = 128
            architecture = "mlp"
            state_dict = checkpoint

        self.class_names = list(class_names)
        self.model = build_model(
            architecture=architecture,
            hidden_size=hidden_size,
            num_classes=len(self.class_names),
        ).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.transform = get_base_transform()

    def predict(self, char_img):
        """Recebe um recorte 28x28 e retorna o simbolo previsto."""
        pil_image = Image.fromarray(char_img)
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        return self.class_names[predicted.item()]
