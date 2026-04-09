import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from src.models.mlp_model import MathMLP

class ModelTrainer:
    """
    Classe responsável pelo treinamento do modelo MLP.
    
    Aplica o princípio de responsabilidade única (SRP) ao gerenciar 
    o ciclo de vida do treinamento.
    """
    def __init__(self, 
                 epochs: int = 5, 
                 batch_size: int = 64, 
                 learning_rate: float = 0.001,
                 device: Optional[str] = None):
        """
        Inicializa o treinador com hiperparâmetros.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Instancia o modelo e move para o dispositivo correto
        self.model = MathMLP().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _get_dataloaders(self, data_dir: str | Path) -> DataLoader:
        """
        Prepara e retorna o DataLoader para o dataset MNIST.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        dataset = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, data_dir: str | Path, save_path: str | Path) -> None:
        """
        Executa o loop de treinamento e salva os pesos finais.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        train_loader = self._get_dataloaders(data_dir)
        
        print(f"[INFO] Iniciando treinamento em: {self.device}")
        self.model.train()
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            loop = tqdm(train_loader, leave=True)
            
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                loop.set_description(f"Epoch [{epoch+1}/{self.epochs}]")
                loop.set_postfix(loss=loss.item())

        # Salva o estado do modelo
        torch.save(self.model.state_dict(), save_path)
        print(f"\n[SUCESSO] Treino concluído! Pesos salvos em: {save_path}")

def main():
    """
    Ponto de entrada para execução do treinamento.
    """
    # Define caminhos relativos à raiz do projeto
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    save_path = base_dir / "models" / "math_mlp_weights.pth"
    
    trainer = ModelTrainer(epochs=5, batch_size=64, learning_rate=0.001)
    trainer.train(data_dir, save_path)

if __name__ == "__main__":
    main()