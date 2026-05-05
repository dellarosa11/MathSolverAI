from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from main import MathSolverAI


LABEL_FILENAME_MAP = {
    "+": "plus",
    "-": "minus",
    "*": "times",
    "/": "div",
    "=": "equals",
    "(": "lparen",
    ")": "rparen",
}


def slugify_label(label: str) -> str:
    if label in LABEL_FILENAME_MAP:
        return LABEL_FILENAME_MAP[label]

    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_").lower()
    return sanitized or "symbol"


def _save_annotated_image(output_path: Path, original_image, symbols) -> None:
    if len(original_image.shape) == 2:
        canvas = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = original_image.copy()

    for index, symbol in enumerate(symbols, start=1):
        x, y, w, h = symbol.box
        label = f"{index}:{symbol.label} {symbol.confidence * 100:.1f}%"
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            label,
            (x, max(12, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exporta recortes e metadados da inferencia do MathSolverAI para inspecao manual.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Imagem que sera analisada.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Checkpoint treinado usado na inferencia.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "debug_outputs" / "inference",
        help="Diretorio onde os recortes e o resumo JSON serao salvos.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Quantidade de candidatos mantidos no resumo por simbolo.",
    )
    parser.add_argument(
        "--disable-correction",
        action="store_true",
        help="Desliga o corretor que testa alternativas do top-k.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=24,
        help="Largura do beam search do corretor.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="Quantidade maxima de candidatos mantidos no resumo do corretor.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    image_path, model_path = MathSolverAI.validate_paths(args.image, args.model)
    app = MathSolverAI(model_path)

    original_image, binary_image = app.processor.get_processed_pipeline(image_path)
    recognition = app.recognize_expression(image_path, top_k=args.top_k)
    correction = app.improve_expression(
        recognition,
        use_correction=not args.disable_correction,
        beam_width=args.beam_width,
        alternatives_per_symbol=args.top_k,
        max_candidates=args.max_candidates,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_annotated_image(args.output_dir / "annotated_predictions.png", original_image, recognition.symbols)

    symbols_payload: list[dict] = []
    for index, symbol in enumerate(recognition.symbols, start=1):
        x, y, w, h = symbol.box
        raw_crop = binary_image[y:y + h, x:x + w]
        nn_crop = app.processor.prepare_for_nn(raw_crop)

        safe_label = slugify_label(symbol.label)
        raw_filename = f"{index:02d}_{safe_label}_raw.png"
        nn_filename = f"{index:02d}_{safe_label}_nn.png"

        cv2.imwrite(str(args.output_dir / raw_filename), raw_crop)
        cv2.imwrite(str(args.output_dir / nn_filename), nn_crop)

        symbols_payload.append(
            {
                "index": index,
                "label": symbol.label,
                "confidence": symbol.confidence,
                "box": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                },
                "raw_crop_file": raw_filename,
                "nn_crop_file": nn_filename,
                "top_predictions": symbol.top_predictions,
            }
        )

    summary = {
        "image_path": str(image_path),
        "model_path": str(model_path),
        "recognized_expression": recognition.expression,
        "corrected_expression": correction.corrected_expression,
        "correction_changed_expression": correction.changed,
        "correction_candidates": [asdict(candidate) for candidate in correction.candidates],
        "symbol_count": len(recognition.symbols),
        "annotated_image": "annotated_predictions.png",
        "symbols": symbols_payload,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] Expressao reconhecida: {recognition.expression}")
    print(f"[INFO] Expressao corrigida: {correction.corrected_expression}")
    print(f"[INFO] Resumo salvo em: {summary_path}")
    print(f"[SUCESSO] Arquivos de depuracao salvos em: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
