from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data.class_config import get_folder_name_for_label


def load_debug_summary(summary_path: str | Path) -> dict[str, Any]:
    path = Path(summary_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _labels_from_corrections_file(corrections_path: Path) -> list[str]:
    payload = json.loads(corrections_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "labels" in payload:
        return [str(label) for label in payload["labels"]]

    if isinstance(payload, dict) and "symbols" in payload:
        ordered_symbols = sorted(payload["symbols"], key=lambda item: int(item["index"]))
        return [str(item["label"]) for item in ordered_symbols]

    raise ValueError("Arquivo de correcoes invalido. Use {'labels': [...]} ou {'symbols': [...]} .")


def resolve_target_labels(
    summary: dict[str, Any],
    *,
    expected_expression: str | None = None,
    corrections_path: str | Path | None = None,
    use_corrected_expression: bool = False,
) -> list[str]:
    if corrections_path is not None:
        labels = _labels_from_corrections_file(Path(corrections_path))
    elif expected_expression is not None:
        labels = list(expected_expression.replace(" ", ""))
    elif use_corrected_expression:
        corrected_expression = summary.get("corrected_expression")
        if not corrected_expression:
            raise ValueError("O summary nao contem 'corrected_expression'.")
        labels = list(str(corrected_expression))
    else:
        raise ValueError(
            "Informe --expected-expression, --corrections-file ou --use-corrected-expression."
        )

    if len(labels) != len(summary.get("symbols", [])):
        raise ValueError(
            "A quantidade de labels fornecida nao bate com a quantidade de simbolos no summary."
        )
    return labels


def import_corrections(
    summary: dict[str, Any],
    *,
    summary_path: str | Path,
    labels: list[str],
    output_dir: str | Path,
    split: str,
    use_raw_crops: bool = False,
    mismatches_only: bool = False,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    source_dir = Path(summary_path).resolve().parent
    output_root = Path(output_dir) / split
    output_root.mkdir(parents=True, exist_ok=True)

    image_stem = Path(summary.get("image_path", "debug_image")).stem
    imported_files: list[dict[str, Any]] = []

    for symbol, target_label in zip(summary.get("symbols", []), labels):
        predicted_label = str(symbol["label"])
        confidence = float(symbol.get("confidence", 0.0))
        if confidence < min_confidence:
            continue
        if mismatches_only and predicted_label == target_label:
            continue

        crop_key = "raw_crop_file" if use_raw_crops else "nn_crop_file"
        crop_filename = symbol[crop_key]
        source_crop = source_dir / crop_filename
        if not source_crop.exists():
            raise FileNotFoundError(f"Recorte nao encontrado: {source_crop}")

        folder_name = get_folder_name_for_label(target_label)
        destination_dir = output_root / folder_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        index = int(symbol["index"])
        destination_name = f"{image_stem}_s{index:02d}_{folder_name}.png"
        destination_path = destination_dir / destination_name
        counter = 1
        while destination_path.exists():
            destination_name = f"{image_stem}_s{index:02d}_{folder_name}_{counter}.png"
            destination_path = destination_dir / destination_name
            counter += 1

        shutil.copy2(source_crop, destination_path)
        imported_files.append(
            {
                "index": index,
                "predicted_label": predicted_label,
                "target_label": target_label,
                "confidence": confidence,
                "source_crop": str(source_crop),
                "destination": str(destination_path),
            }
        )

    return {
        "image_path": summary.get("image_path"),
        "split": split,
        "imported_count": len(imported_files),
        "files": imported_files,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Importa recortes do debug como novos exemplos rotulados do dataset.",
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Caminho do summary.json gerado por export_inference_debug.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "symbols",
        help="Diretorio base do dataset customizado.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Split de destino.",
    )
    parser.add_argument(
        "--expected-expression",
        default="",
        help="Expressao correta para mapear cada simbolo na ordem detectada.",
    )
    parser.add_argument(
        "--corrections-file",
        default="",
        help="Arquivo JSON com labels corrigidos por indice.",
    )
    parser.add_argument(
        "--use-corrected-expression",
        action="store_true",
        help="Usa 'corrected_expression' do summary como fonte de rotulos.",
    )
    parser.add_argument(
        "--use-raw-crops",
        action="store_true",
        help="Importa os recortes crus em vez dos recortes preparados para a rede.",
    )
    parser.add_argument(
        "--mismatches-only",
        action="store_true",
        help="Importa apenas simbolos cujo rotulo previsto difere do rotulo alvo.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Importa apenas simbolos com confianca minima informada.",
    )
    parser.add_argument(
        "--manifest-path",
        default="",
        help="Caminho opcional para salvar um resumo JSON da importacao.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    summary = load_debug_summary(args.summary)
    labels = resolve_target_labels(
        summary,
        expected_expression=args.expected_expression or None,
        corrections_path=args.corrections_file or None,
        use_corrected_expression=args.use_corrected_expression,
    )
    report = import_corrections(
        summary,
        summary_path=args.summary,
        labels=labels,
        output_dir=args.output_dir,
        split=args.split,
        use_raw_crops=args.use_raw_crops,
        mismatches_only=args.mismatches_only,
        min_confidence=args.min_confidence,
    )

    print(f"[INFO] Simbolos importados: {report['imported_count']}")
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[INFO] Manifesto salvo em: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
