import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Configuração do path para encontrar o pacote src
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.preprocessor import ImageProcessor

def test_vision_module():
    """Testa o módulo de visão computacional."""
    image_path = BASE_DIR / "data" / "raw" / "teste.jpg"
    
    try:
        # Instancia o objeto (Conceito de POO)
        processor = ImageProcessor(image_path)
        
        # Executa a lógica encapsulada
        original, processed, symbols = processor.get_processed_pipeline()
        
        # Visualização (Apenas para conferência humana)
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(processed, cmap='gray')
        plt.title("Processada")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"[SUCCESS] Image Processor Work. {len(symbols)} symbols detected.")
        
    except Exception as e:
        print(f"[FALHA] Exceção capturada: {e}")

if __name__ == "__main__":
    test_vision_module()