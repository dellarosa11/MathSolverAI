import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List


class ImageProcessor:

    def __init__(self, image_path: str | Path ):
        self.image_path = str(image_path)
        self.original_image = None
        
        # 1 - Gray Scale
    def load_image(self) -> np.ndarray:     # Load image in GrayScale
        self.original_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE) 
        if self.original_image is None:
            raise FileNotFoundError(f"Imagem não encontrada! Path: {self.image_path}")
        return self.original_image
    
    def get_processed_pipeline(self) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        #                                   OriginalImg, ProcessedImg, SymbolList
        
        image = self.load_image()
        
        # 2 - Blurring
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 3 - Binarization (Black and White)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        # 4 - Bold
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        # 5 - Find Outlines
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        symbols = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            #size filter to ignore noise 
            if w > 5 and h > 5:
                roi = thresh[y:y+h, x:x+w]
                symbols.append((x, roi)) # X to order Left to Right
        # Order with X Lto R
        symbols.sort(key=lambda item: item[0])
        ordered_symbols = [s[1] for s in symbols]
        return image, thresh, ordered_symbols
    

# --- Bloco de Teste ---
if __name__ == "__main__":
    # Substitua pelo caminho de uma foto que você tirar de um papel escrito "2+2"
    # test_path = "data/raw/teste_equacao.jpg" 
    # thresh_img, detected_symbols = process_equation_image(test_path)
    print("Script do PreProcesser pronto!")