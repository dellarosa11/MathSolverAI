import torch.nn as nn
import torch.nn.functional as F

class MathMLP(nn.Module):
    """
    Arquitetura MLP (Multi-Layer Perceptron) para reconhecimento de caracteres matemáticos.
    
    Esta classe define uma rede neural simples com camadas totalmente conectadas,
    seguindo o princípio de responsabilidade única (SRP) ao focar apenas na 
    definição da arquitetura.

    Entrada: Imagem 28x28 (784 pixels)
    Saída: Probabilidade de ser um dos 10 dígitos (0-9) pois falta implementar outros datasets
    """
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        """
        Inicializa as camadas da rede neural.
        Args:
            input_size (int): Tamanho do vetor de entrada (padrão 28x28 = 784).
            hidden_size (int): Número de neurônios na primeira camada oculta.
            num_classes (int): Número de classes de saída (padrão 10 para dígitos 0-9).
        """
        super(MathMLP, self).__init__()
        # Camada de entrada -> Escondida 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Camada escondida 1 -> Escondida 2
        self.fc2 = nn.Linear(hidden_size, 64)
        # Camada de saída
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Define o fluxo de dados (forward pass) da rede.
        """
        # Achata a imagem (28, 28) -> (784)
        x = x.view(-1, 784)

        # Funções de ativação ReLU (ajudam a rede a aprender padrões complexos)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Retorna os logits (valores brutos antes da ativação final)
        return self.fc3(x)