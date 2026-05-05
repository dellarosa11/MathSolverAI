from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data.class_config import get_folder_name_for_label
from src.preprocessor import ImageProcessor


TOKEN_NORMALIZATION: Dict[str, str] = {
    r"\times": "*",
    r"\div": "/",
}

ALLOWED_LABELS = {
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "+",
    "-",
    "*",
    "/",
    "=",
    "(",
    ")",
}

BLOCKED_TOKENS = {
    "{",
    "}",
    "^",
    "_",
    ",",
    ".",
    "[",
    "]",
    "|",
}


@dataclass
class SampleRecord:
    image_path: Path
    labels: List[str]
    source_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filtra expressoes simples rotuladas e exporta simbolos segmentados por classe.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Pasta raiz do dataset com imagens e arquivos caption.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "symbols",
        help="Diretorio de destino com subpastas train/val.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Percentual reservado para validacao.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limite total de expressoes simples a processar. Use 0 para sem limite.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=9,
        help="Numero maximo de simbolos aceitos por expressao simples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para embaralhamento reprodutivel.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove o conteudo atual de output-dir antes de exportar.",
    )
    return parser.parse_args()


def normalize_token(token: str) -> str | None:
    token = TOKEN_NORMALIZATION.get(token, token)
    if token in BLOCKED_TOKENS:
        return None
    return token


def extract_labels(caption: str, max_tokens: int) -> List[str] | None:
    raw_tokens = [part.strip() for part in caption.split() if part.strip()]
    labels: List[str] = []

    for token in raw_tokens:
        normalized = normalize_token(token)
        if normalized is None:
            return None
        if normalized not in ALLOWED_LABELS:
            return None
        labels.append(normalized)

    if not labels or len(labels) > max_tokens:
        return None

    return labels


def load_candidate_records(source_dir: Path, max_tokens: int) -> List[SampleRecord]:
    records: List[SampleRecord] = []

    for caption_path in sorted(source_dir.rglob("caption.txt")):
        image_root = caption_path.parent
        with caption_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split("\t", 1)
                if len(parts) != 2:
                    continue

                image_stem, caption = parts
                labels = extract_labels(caption, max_tokens=max_tokens)
                if labels is None:
                    continue

                image_path = image_root / f"{image_stem}.bmp"
                if not image_path.exists():
                    continue

                records.append(
                    SampleRecord(
                        image_path=image_path,
                        labels=labels,
                        source_name=caption_path.parent.name,
                    )
                )

    return records


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs(output_dir: Path) -> None:
    for split in ("train", "val"):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for label in sorted(ALLOWED_LABELS):
            folder_name = get_folder_name_for_label(label)
            (split_dir / folder_name).mkdir(parents=True, exist_ok=True)


def split_records(records: Sequence[SampleRecord], val_ratio: float, seed: int) -> tuple[List[SampleRecord], List[SampleRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    val_count = int(len(shuffled) * val_ratio)
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]
    return train_records, val_records


def export_records(records: Iterable[SampleRecord], split: str, output_dir: Path) -> Dict[str, int]:
    processor = ImageProcessor()
    counters: Dict[str, int] = {label: 0 for label in ALLOWED_LABELS}

    for record_index, record in enumerate(records):
        _, binary = processor.get_processed_pipeline(record.image_path)
        boxes = processor.extract_bounding_boxes(binary)

        if len(boxes) != len(record.labels):
            continue

        for symbol_index, ((x, y, w, h), label) in enumerate(zip(boxes, record.labels)):
            roi = binary[y:y + h, x:x + w]
            symbol_img = processor.prepare_for_nn(roi)

            folder_name = get_folder_name_for_label(label)
            counters[label] += 1
            file_name = f"{record.image_path.stem}_{record_index}_{symbol_index}.png"
            save_path = output_dir / split / folder_name / file_name
            cv2.imwrite(str(save_path), symbol_img)

    return counters


def print_summary(train_counts: Dict[str, int], val_counts: Dict[str, int]) -> None:
    print("[INFO] Exportacao concluida.")
    print("[INFO] Quantidade de simbolos por classe:")
    for label in sorted(ALLOWED_LABELS):
        print(
            f"  {label}: "
            f"train={train_counts.get(label, 0)} "
            f"val={val_counts.get(label, 0)}"
        )


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Pasta de origem nao encontrada: {source_dir}")

    print(f"[INFO] Lendo captions em: {source_dir}")
    records = load_candidate_records(source_dir, max_tokens=args.max_tokens)
    print(f"[INFO] Expressoes simples candidatas encontradas: {len(records)}")

    if args.max_samples > 0:
        records = records[:args.max_samples]
        print(f"[INFO] Limite aplicado. Expressoes selecionadas: {len(records)}")

    if not records:
        print("[AVISO] Nenhuma expressao simples elegivel foi encontrada.")
        return

    if args.clean_output:
        reset_output_dir(output_dir)

    ensure_output_dirs(output_dir)
    train_records, val_records = split_records(records, args.val_ratio, args.seed)
    print(f"[INFO] Expressoes de treino: {len(train_records)}")
    print(f"[INFO] Expressoes de validacao: {len(val_records)}")

    train_counts = export_records(train_records, split="train", output_dir=output_dir)
    val_counts = export_records(val_records, split="val", output_dir=output_dir)
    print_summary(train_counts, val_counts)


if __name__ == "__main__":
    main()
