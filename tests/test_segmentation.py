import sys
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Adiciona o diretório raiz ao sys.path para encontrar o pacote src
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.preprocessor import ImageProcessor

def test_segmentation_visual():
    """
    Testa visualmente a segmentação e o pré-processamento para redes neurais.
    """
    # Caminho da imagem de teste
    image_path = BASE_DIR / "data" / "raw" / "teste.jpg"
    
    if not image_path.exists():
        print(f"[ERRO] Imagem não encontrada em: {image_path}")
        return

    try:
        # 1. Instancia e executa o pipeline básico
        processor = ImageProcessor(image_path)
        original, binary = processor.get_processed_pipeline()
        
        # 2. Extrai as bounding boxes (Desacoplado)
        boxes = processor.extract_bounding_boxes(binary)
        print(f"[INFO] {len(boxes)} símbolos detectados.")

        # 3. Prepara a imagem para visualização dos retângulos
        # Converte para BGR para desenhar em cores
        vis_original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        processed_symbols = []
        for (x, y, w, h) in boxes:
            # Desenha retângulo verde na imagem original
            cv2.rectangle(vis_original, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extrai a ROI da imagem binária
            roi = binary[y:y+h, x:x+w]
            
            # Prepara para Rede Neural (Pad & Resize)
            nn_input = processor.prepare_for_nn(roi, target_size=28)
            processed_symbols.append(nn_input)

        # 4. Visualização com Matplotlib
        plt.figure(figsize=(15, 8))
        
        # Subplot 1: Imagem Original com Bounding Boxes
        plt.subplot(2, 1, 1)
        plt.imshow(cv2.cvtColor(vis_original, cv2.COLOR_BGR2RGB))
        plt.title(f"Detecção de Símbolos ({len(boxes)} encontrados)")
        plt.axis('off')

        # Subplot 2: Símbolos Processados para NN (Lado a Lado)
        if processed_symbols:
            # Cria uma grade para os símbolos
            num_symbols = len(processed_symbols)
            for i, sym in enumerate(processed_symbols):
                plt.subplot(2, num_symbols, num_symbols + i + 1)
                plt.imshow(sym, cmap='gray')
                plt.title(f"S{i+1}")
                plt.axis('off')

        plt.tight_layout()
        plt.show()
        print("[SUCCESS] Teste de segmentação concluído com sucesso.")

    except Exception as e:
        print(f"[FALHA] Ocorreu um erro durante o teste: {e}")

if __name__ == "__main__":
    test_segmentation_visual()