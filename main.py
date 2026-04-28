from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Adiciona a raiz do projeto ao path para garantir que os imports funcionem
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.models.predictor import MathPredictor, PredictionResult
from src.postprocessor import ExpressionCorrectionResult, ExpressionCorrector
from src.preprocessor import ImageProcessor
from src.solver import MathSolver


@dataclass(frozen=True)
class RecognizedSymbol:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    top_predictions: list[dict[str, float]]


@dataclass(frozen=True)
class RecognitionResult:
    expression: str
    symbols: list[RecognizedSymbol]


class MathSolverAI:
    """
    Classe principal que coordena o pipeline completo do projeto MathSolverAI.
    """

    def __init__(self, model_path: str | Path):
        self.processor = ImageProcessor()
        self.predictor = MathPredictor(model_path)
        self.solver = MathSolver()
        self.corrector = ExpressionCorrector(self.solver)

    @staticmethod
    def validate_paths(image_path: str | Path, model_path: str | Path) -> tuple[Path, Path]:
        resolved_image = Path(image_path)
        resolved_model = Path(model_path)

        if not resolved_model.exists():
            raise FileNotFoundError(
                f"Modelo nao encontrado em: {resolved_model}. "
                "Execute 'python src/train.py' para gerar os pesos."
            )

        if not resolved_image.exists():
            raise FileNotFoundError(f"Imagem de entrada nao encontrada em: {resolved_image}")

        return resolved_image, resolved_model

    def _predict_symbol(self, char_img, top_k: int) -> PredictionResult:
        if hasattr(self.predictor, "predict_with_confidence"):
            return self.predictor.predict_with_confidence(char_img, top_k=top_k)

        label = str(self.predictor.predict(char_img))
        return PredictionResult(
            label=label,
            confidence=1.0,
            top_predictions=[{"label": label, "confidence": 1.0}],
        )

    def recognize_expression(self, image_path: str | Path, top_k: int = 3) -> RecognitionResult:
        print(f"[INFO] Processando imagem: {image_path}")
        _, binary = self.processor.get_processed_pipeline(image_path)
        boxes = self.processor.extract_bounding_boxes(binary)
        sort_boxes = getattr(
            self.processor,
            "sort_boxes_reading_order",
            lambda items: sorted(items, key=lambda item: (item[1], item[0])),
        )
        boxes = sort_boxes(list(boxes))

        if not boxes:
            print("[AVISO] Nenhum caractere detectado na imagem.")
            return RecognitionResult(expression="", symbols=[])

        recognized_symbols: list[RecognizedSymbol] = []
        for x, y, w, h in boxes:
            roi = binary[y:y + h, x:x + w]
            char_img = self.processor.prepare_for_nn(roi)
            prediction = self._predict_symbol(char_img, top_k=top_k)
            recognized_symbols.append(
                RecognizedSymbol(
                    label=prediction.label,
                    confidence=prediction.confidence,
                    box=(x, y, w, h),
                    top_predictions=prediction.top_predictions,
                )
            )

        expression = "".join(symbol.label for symbol in recognized_symbols)
        print(f"[INFO] Equacao reconhecida: {expression}")
        return RecognitionResult(expression=expression, symbols=recognized_symbols)

    @staticmethod
    def print_diagnostics(
        recognition: RecognitionResult,
        correction: ExpressionCorrectionResult | None = None,
    ) -> None:
        if not recognition.symbols:
            print("[INFO] Nenhum simbolo para diagnosticar.")
            return

        print("[INFO] Diagnostico da inferencia:")
        for index, symbol in enumerate(recognition.symbols, start=1):
            x, y, w, h = symbol.box
            candidates = ", ".join(
                f"{candidate['label']} ({candidate['confidence'] * 100:.1f}%)"
                for candidate in symbol.top_predictions
            )
            print(
                f"  S{index}: label={symbol.label} "
                f"confidence={symbol.confidence * 100:.1f}% "
                f"box=({x}, {y}, {w}, {h}) "
                f"top={candidates}"
            )

        if correction is not None:
            print("[INFO] Candidatos apos o corretor:")
            for index, candidate in enumerate(correction.candidates, start=1):
                print(
                    f"  C{index}: expr={candidate.expression} "
                    f"score={candidate.score:.3f} "
                    f"valid={'sim' if candidate.valid else 'nao'} "
                    f"solvable={'sim' if candidate.solvable else 'nao'}"
                )

    def improve_expression(
        self,
        recognition: RecognitionResult,
        *,
        use_correction: bool = True,
        beam_width: int = 24,
        alternatives_per_symbol: int = 3,
        max_candidates: int = 8,
    ) -> ExpressionCorrectionResult:
        if not use_correction:
            return self.corrector.identity(recognition.expression)

        correction = self.corrector.correct(
            recognition,
            beam_width=beam_width,
            alternatives_per_symbol=alternatives_per_symbol,
            max_candidates=max_candidates,
        )
        if correction.changed:
            print(
                f"[INFO] Corretor sugeriu: {correction.corrected_expression} "
                f"(antes: {correction.raw_expression})"
            )
        return correction

    def run_pipeline(
        self,
        image_path: str | Path,
        *,
        diagnostic: bool = False,
        solve_expression: bool = True,
        top_k: int = 3,
        use_correction: bool = True,
        beam_width: int = 24,
        max_candidates: int = 8,
    ) -> str:
        recognition = self.recognize_expression(image_path, top_k=top_k)
        if not recognition.expression:
            return ""

        correction = self.improve_expression(
            recognition,
            use_correction=use_correction,
            beam_width=beam_width,
            alternatives_per_symbol=top_k,
            max_candidates=max_candidates,
        )
        expression_to_use = correction.corrected_expression

        if diagnostic:
            self.print_diagnostics(recognition, correction)

        if not solve_expression:
            return expression_to_use

        try:
            normalized_expression = self.solver.normalize_expression(expression_to_use)
            if normalized_expression != expression_to_use:
                print(f"[INFO] Expressao normalizada: {normalized_expression}")

            result = self.solver.solve(normalized_expression)
            print(f"[SUCESSO] Resultado: {result}")
            return str(result)
        except Exception as exc:
            print(f"[ERRO] Falha ao resolver a equacao: {exc}")
            return f"Erro: {exc}"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa o pipeline de reconhecimento e resolucao de expressoes matematicas.",
    )
    parser.add_argument(
        "--image",
        default=str(BASE_DIR / "data" / "raw" / "teste.jpg"),
        help="Caminho da imagem contendo a expressao matematica.",
    )
    parser.add_argument(
        "--model",
        default=str(BASE_DIR / "models" / "math_mlp_weights.pth"),
        help="Caminho do checkpoint treinado.",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Mostra confianca e alternativas de predicao por simbolo.",
    )
    parser.add_argument(
        "--recognize-only",
        action="store_true",
        help="Reconhece a expressao, mas nao tenta resolve-la.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Quantidade de candidatos mostrados no modo diagnostico.",
    )
    parser.add_argument(
        "--disable-correction",
        action="store_true",
        help="Desliga o corretor que usa alternativas do top-k.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=24,
        help="Largura do beam search usado pelo corretor.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="Quantidade maxima de candidatos exibidos no corretor.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        image_path, model_path = MathSolverAI.validate_paths(args.image, args.model)
    except FileNotFoundError as exc:
        print(f"[ERRO] {exc}")
        return 1

    app = MathSolverAI(model_path)
    app.run_pipeline(
        image_path,
        diagnostic=args.diagnostic,
        solve_expression=not args.recognize_only,
        top_k=args.top_k,
        use_correction=not args.disable_correction,
        beam_width=args.beam_width,
        max_candidates=args.max_candidates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
