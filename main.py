import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Adiciona a raiz do projeto ao path para garantir que os imports funcionem
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.preprocessor import ImageProcessor
from src.models.predictor import MathPredictor
from src.solver import MathSolver

class MathSolverAI:
    """
    Classe principal que coordena o pipeline completo do projeto MathSolverAI.
    
    Aplica o princípio de composição (Composição sobre Herança) ao 
    integrar o processador de imagem, o preditor e o resolvedor.
    """
    def __init__(self, model_path: str | Path):
        """
        Inicializa o pipeline com os componentes necessários.
        
        Args:
            model_path (str | Path): Caminho para o arquivo de pesos do modelo.
        """
        self.processor = ImageProcessor()
        self.predictor = MathPredictor(model_path)
        self.solver = MathSolver()

    def run_pipeline(self, image_path: str | Path) -> str:
        """
        Executa o pipeline completo: Processamento -> Reconhecimento -> Resolução.
        
        Args:
            image_path (str | Path): Caminho para a imagem da equação.
            
        Returns:
            str: O resultado da resolução da equação.
        """
        print(f"[INFO] Processando imagem: {image_path}")
        
        # 1. Processamento de Imagem
        original, binary = self.processor.get_processed_pipeline(image_path)
        boxes = self.processor.extract_bounding_boxes(binary)
        
        if not boxes:
            print("[AVISO] Nenhum caractere detectado na imagem.")
            return ""
        
        # 2. Reconhecimento de Caracteres
        equation_str = ""
        for (x, y, w, h) in boxes:
            # Recorta a Região de Interesse (ROI)
            roi = binary[y:y+h, x:x+w]
            
            # Prepara para a rede neural (28x28, centralizado)
            char_img = self.processor.prepare_for_nn(roi)
            
            # Realiza a predição
            detected_char = self.predictor.predict(char_img)
            equation_str += str(detected_char)
        
        print(f"[INFO] Equação reconhecida: {equation_str}")
        
        # 3. Resolução da Equação (Próximo passo: integrar reconhecimento de operadores)
        # Por enquanto, resolvemos o que foi detectado
        try:
            result = self.solver.solve(equation_str)
            print(f"[SUCESSO] Resultado: {result}")
            return str(result)
        except Exception as e:
            print(f"[ERRO] Falha ao resolver a equação: {e}")
            return f"Erro: {e}"

def main():
    """
    Ponto de entrada principal da aplicação.
    """
    # Caminhos padrão
    model_path = BASE_DIR / "models" / "math_mlp_weights.pth"
    test_image = BASE_DIR / "data" / "raw" / "teste.jpg"
    
    # Verifica se os arquivos necessários existem
    if not model_path.exists():
        print(f"[ERRO] Modelo não encontrado em: {model_path}")
        print("Por favor, execute 'python src/train.py' primeiro para treinar o modelo.")
        return

    if not test_image.exists():
        print(f"[ERRO] Imagem de teste não encontrada em: {test_image}")
        return

    # Instancia e executa o pipeline
    app = MathSolverAI(model_path)
    app.run_pipeline(test_image)

if __name__ == "__main__":
    main()