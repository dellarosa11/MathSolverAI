import torch
import torch.nn.functional as F
from torchvision import transforms
import sys
from pathlib import Path

# Adiciona a raiz do projeto (duas pastas acima de predictor.py) ao path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.models.mlp_model import MathMLP

class MathPredictor:
    
    """
    Classe responsável por realizar a inferência (reconhecimento).
    Encapsula o modelo PyTorch e as transformações de dados.
    """
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MathMLP().to(self.device)
        
        # Carrega os pesos treinados
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Modo de avaliação (desliga dropout, etc)
        
        # As transformações DEVEM ser as mesmas do treino
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, char_img):
        """Recebe um recorte 28x28 e retorna o número (0-9)."""
        # Prepara a imagem para o PyTorch (adiciona dimensão de batch)
        img_tensor = self.transform(char_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.item()