import sys
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Setup de Path para encontrar o pacote 'src'
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.preprocessor import ImageProcessor
from src.models.predictor import MathPredictor

def visualize_results():
    # 1. Caminhos
    image_path = BASE_DIR / "data" / "raw" / "teste.jpg"
    model_path = BASE_DIR / "models" / "math_mlp_weights.pth"
    
    # 2. Inicializar Componentes (APOO)
    processor = ImageProcessor(image_path)
    predictor = MathPredictor(model_path)
    
    # 3. Pipeline de Imagem
    original, binary = processor.get_processed_pipeline()
    boxes = processor.extract_bounding_boxes(binary)
    
    # Converter original para cor para desenhar o texto colorido
    output_img = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    
    print(f"Iniciando reconhecimento de {len(boxes)} símbolos...")

    for (x, y, w, h) in boxes:
        # Extrair e preparar o caractere
        roi = binary[y:y+h, x:x+w]
        nn_input = processor.prepare_for_nn(roi)
        
        # PREDICÃO: O MLP analisa a imagem aqui
        label = predictor.predict(nn_input)
        
        # 4. Desenhar na imagem de visualização
        # Retângulo verde
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Texto com o número reconhecido (em azul)
        cv2.putText(output_img, str(label), (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 5. Exibir resultado final
    plt.figure(figsize=(10, 6))
    plt.imshow(output_img)
    plt.title("Resultado da Classificação MLP")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    visualize_results()