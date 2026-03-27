import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List


class ImageProcessor:
    """
    Classe responsável pelo pré-processamento de imagens contendo equações matemáticas.
    """

    def __init__(self, image_path: str | Path):
        self.image_path = str(image_path)
        self.original_image: np.ndarray | None = None
        self.processed_image: np.ndarray | None = None
        
    def load_image(self) -> np.ndarray:
        """Carrega a imagem em escala de cinza."""
        path = Path(self.image_path)
        if not path.exists():
            raise FileNotFoundError(f"O arquivo não existe: {path.absolute()}")

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Falha ao carregar a imagem (formato inválido?): {path.absolute()}")
        
        self.original_image = image
        return self.original_image
    
    def get_processed_pipeline(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executa o pipeline de pré-processamento básico (carregar, blur, binarizar, dilatar).
        Retorna: (imagem_original, imagem_binaria)
        """
        image = self.load_image()
        
        # 2 - Blurring
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 3 - Binarization (Black and White)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 4 - Bold (Dilatação)
        kernel = np.ones((3, 3), np.uint8)
        self.processed_image = cv2.dilate(thresh, kernel, iterations=1)
        
        # Garantia de retorno
        if self.original_image is None or self.processed_image is None:
            raise RuntimeError("Falha ao processar as imagens no pipeline.")

        return self.original_image, self.processed_image
    
    def extract_bounding_boxes(self, binary_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta contornos na imagem binária e retorna as coordenadas das bounding boxes.
        """
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filtro de tamanho para ignorar ruídos
            if w > 5 and h > 5:
                boxes.append((x, y, w, h))

        # Ordenar da esquerda para a direita
        boxes.sort(key=lambda item: item[0])
        return boxes
    
    def prepare_for_nn(self, roi: np.ndarray, target_size: int = 28) -> np.ndarray:
        """
        Prepara uma Região de Interesse (ROI) para entrada em uma Rede Neural.
        """
        h, w = roi.shape
        if h == 0 or w == 0:
            return np.zeros((target_size, target_size), dtype=np.uint8)
        
        # Calcula a escala mantendo a proporção
        if h > w:
            scale = target_size / h
            new_h, new_w = target_size, max(1, int(w * scale))
        else:
            scale = target_size / w
            new_h, new_w = max(1, int(h * scale)), target_size
        
        # Redimensiona
        resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Cria fundo preto e centraliza a ROI
        padded_img = np.zeros((target_size, target_size), dtype=np.uint8)
        start_y = (target_size - new_h) // 2
        start_x = (target_size - new_w) // 2
        
        padded_img[start_y:start_y + new_h, start_x:start_x + new_w] = resized_roi

        return padded_img


if __name__ == "__main__":
    print("ImageProcessor pronto para uso!")