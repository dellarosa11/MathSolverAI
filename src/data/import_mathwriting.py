from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data.class_config import DIGIT_CLASSES, get_folder_name_for_label


INKML_NAMESPACE = {"inkml": "http://www.w3.org/2003/InkML"}
DEFAULT_ARCHIVE_URL = "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz"
DEFAULT_EXCERPT_URL = "https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz"
DEFAULT_ARCHIVE_PATH = BASE_DIR / "data" / "external" / "mathwriting" / "mathwriting-2024.tgz"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "symbols"
DEFAULT_MANIFEST_PATH = BASE_DIR / "data" / "external" / "mathwriting" / "import_manifest.json"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_SEED = 42

LABEL_NORMALIZATION: Dict[str, str] = {
    **{digit: digit for digit in DIGIT_CLASSES},
    "+": "+",
    "-": "-",
    "=": "=",
    "(": "(",
    ")": ")",
    "/": "/",
    r"\div": "/",
    r"\times": "*",
    r"\lparen": "(",
    r"\rparen": ")",
    r"\left(": "(",
    r"\right)": ")",
}
DEFAULT_SELECTED_FOLDERS = tuple(DIGIT_CLASSES + ["plus", "minus", "times", "div", "equals", "lparen", "rparen"])
OPERATORS_ONLY = ("plus", "minus", "times", "div", "equals", "lparen", "rparen")


def download_mathwriting_archive(
    archive_path: Path,
    archive_url: str,
    force_download: bool = False,
) -> Path:
    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if archive_path.exists() and not force_download:
        print(f"[INFO] Reutilizando arquivo ja baixado: {archive_path}")
        return archive_path

    print(f"[INFO] Baixando MathWriting de: {archive_url}")
    with urllib.request.urlopen(archive_url) as response, archive_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)

    print(f"[SUCESSO] Arquivo salvo em: {archive_path}")
    return archive_path


def expand_requested_folders(requested: Sequence[str]) -> list[str]:
    if not requested:
        return list(DEFAULT_SELECTED_FOLDERS)

    expanded: list[str] = []
    for item in requested:
        normalized = item.strip().lower()
        if not normalized:
            continue
        if normalized == "all":
            expanded.extend(DEFAULT_SELECTED_FOLDERS)
            continue
        if normalized == "digits":
            expanded.extend(DIGIT_CLASSES)
            continue
        if normalized == "operators":
            expanded.extend(OPERATORS_ONLY)
            continue
        if normalized in DEFAULT_SELECTED_FOLDERS:
            expanded.append(normalized)
            continue

        try:
            folder_name = get_folder_name_for_label(item)
        except KeyError as exc:
            allowed = ", ".join(DEFAULT_SELECTED_FOLDERS)
            raise ValueError(
                f"Classe MathWriting nao suportada: {item}. Use digits, operators, all ou uma destas classes: {allowed}."
            ) from exc

        if folder_name not in DEFAULT_SELECTED_FOLDERS:
            allowed = ", ".join(DEFAULT_SELECTED_FOLDERS)
            raise ValueError(
                f"O importador do MathWriting nao cobre a classe '{item}'. "
                f"Classes disponiveis: {allowed}."
            )
        expanded.append(folder_name)

    deduplicated: list[str] = []
    seen: set[str] = set()
    for folder_name in expanded:
        if folder_name not in seen:
            deduplicated.append(folder_name)
            seen.add(folder_name)
    return deduplicated


def build_normalized_label_to_folder_mapping(selected_folders: Sequence[str]) -> dict[str, str]:
    selected_set = set(selected_folders)
    mapping: dict[str, str] = {}
    for raw_label, canonical_label in LABEL_NORMALIZATION.items():
        try:
            folder_name = get_folder_name_for_label(canonical_label)
        except KeyError:
            continue
        if folder_name in selected_set:
            mapping[raw_label] = folder_name
    return mapping


def _extract_annotation(root: ET.Element, annotation_type: str) -> str | None:
    for annotation in root.findall("inkml:annotation", INKML_NAMESPACE):
        if annotation.attrib.get("type") == annotation_type:
            return (annotation.text or "").strip()
    return None


def _parse_trace_points(trace_text: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for raw_point in trace_text.strip().split(","):
        tokens = raw_point.strip().split()
        if len(tokens) < 2:
            continue
        points.append((float(tokens[0]), float(tokens[1])))
    return points


def _load_symbol_entries(
    archive_path: Path,
    label_to_folder: Mapping[str, str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        root_prefix = None
        inside_symbols = False
        for member in archive:
            member_path = member.name
            if root_prefix is None:
                root_prefix = member_path.split("/", 1)[0]
                symbols_prefix = f"{root_prefix}/symbols/"
            if member.isdir():
                continue

            if member_path.startswith(symbols_prefix):
                inside_symbols = True
            elif inside_symbols:
                break
            else:
                continue

            if not member_path.endswith(".inkml"):
                continue

            file_obj = archive.extractfile(member)
            if file_obj is None:
                continue
            root = ET.fromstring(file_obj.read())
            label = _extract_annotation(root, "label")
            if label is None:
                continue
            folder_name = label_to_folder.get(label)
            if folder_name is None:
                continue

            traces = [
                _parse_trace_points(trace.text or "")
                for trace in root.findall("inkml:trace", INKML_NAMESPACE)
            ]
            traces = [trace for trace in traces if trace]
            if not traces:
                continue

            entries.append(
                {
                    "sample_id": _extract_annotation(root, "sampleId") or Path(member_path).stem,
                    "label": label,
                    "folder_name": folder_name,
                    "traces": traces,
                }
            )

    return entries


def _render_traces_to_image(
    traces: Sequence[Sequence[tuple[float, float]]],
    *,
    image_size: int = 28,
    margin: int = 3,
    stroke_width: int = 2,
) -> np.ndarray:
    xs = [x for trace in traces for x, _ in trace]
    ys = [y for trace in traces for _, y in trace]
    if not xs or not ys:
        return np.zeros((image_size, image_size), dtype=np.uint8)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    inner_size = max(4, image_size - (margin * 2))
    scale = inner_size / max(width, height)

    scaled_width = width * scale
    scaled_height = height * scale
    offset_x = (image_size - scaled_width) / 2.0
    offset_y = (image_size - scaled_height) / 2.0

    canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    for trace in traces:
        points = np.array(
            [
                [
                    int(round((x - min_x) * scale + offset_x)),
                    int(round((y - min_y) * scale + offset_y)),
                ]
                for x, y in trace
            ],
            dtype=np.int32,
        )
        if len(points) == 1:
            cv2.circle(canvas, tuple(points[0]), radius=max(1, stroke_width // 2), color=255, thickness=-1)
        else:
            cv2.polylines(canvas, [points], isClosed=False, color=255, thickness=stroke_width, lineType=cv2.LINE_AA)
    return canvas


def build_split_plan(
    entries: Sequence[dict[str, Any]],
    *,
    train_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio deve ficar entre 0 e 1.")

    by_folder: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_folder.setdefault(str(entry["folder_name"]), []).append(entry)

    plan = {"train": [], "val": []}
    for folder_name, folder_entries in by_folder.items():
        shuffled = list(folder_entries)
        random.Random(f"{seed}:{folder_name}").shuffle(shuffled)

        if len(shuffled) <= 1:
            train_count = len(shuffled)
        else:
            train_count = int(round(len(shuffled) * train_ratio))
            train_count = max(1, min(len(shuffled) - 1, train_count))

        plan["train"].extend(shuffled[:train_count])
        plan["val"].extend(shuffled[train_count:])

    return plan


def remove_previous_imports(output_dir: Path, selected_folders: Sequence[str]) -> int:
    removed = 0
    for split in ("train", "val"):
        for folder_name in selected_folders:
            target_dir = output_dir / split / folder_name
            if not target_dir.exists():
                continue
            for image_path in target_dir.glob("mathwriting_*.png"):
                image_path.unlink()
                removed += 1
    return removed


def import_mathwriting_symbols(
    archive_path: Path,
    output_dir: Path,
    *,
    label_to_folder: Mapping[str, str],
    train_ratio: float,
    seed: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    entries = _load_symbol_entries(archive_path, label_to_folder)
    split_plan = build_split_plan(entries, train_ratio=train_ratio, seed=seed)

    folder_names = sorted(set(label_to_folder.values()))
    train_counts = {folder_name: 0 for folder_name in folder_names}
    val_counts = {folder_name: 0 for folder_name in folder_names}
    source_label_counts: dict[str, int] = {}
    train_counts["skipped_existing"] = 0
    val_counts["skipped_existing"] = 0

    for split_name, split_entries in split_plan.items():
        split_counts = train_counts if split_name == "train" else val_counts
        for entry in split_entries:
            folder_name = str(entry["folder_name"])
            destination_dir = output_dir / split_name / folder_name
            destination_dir.mkdir(parents=True, exist_ok=True)

            sample_id = str(entry["sample_id"])
            destination_path = destination_dir / f"mathwriting_{sample_id}.png"
            if destination_path.exists():
                split_counts["skipped_existing"] += 1
                continue

            image = _render_traces_to_image(entry["traces"])
            cv2.imwrite(str(destination_path), image)
            split_counts[folder_name] += 1
            source_label_counts[str(entry["label"])] = source_label_counts.get(str(entry["label"]), 0) + 1

    return train_counts, val_counts, source_label_counts


def write_manifest(
    manifest_path: Path,
    *,
    archive_path: Path,
    archive_url: str,
    train_ratio: float,
    seed: int,
    selected_folders: Sequence[str],
    label_to_folder: Mapping[str, str],
    train_counts: Mapping[str, int],
    val_counts: Mapping[str, int],
    source_label_counts: Mapping[str, int],
    removed_previous_imports: int,
) -> None:
    manifest: dict[str, Any] = {
        "dataset": "MathWriting",
        "archive_path": str(archive_path),
        "archive_url": archive_url,
        "train_ratio": train_ratio,
        "seed": seed,
        "selected_folders": list(selected_folders),
        "source_label_to_folder": dict(label_to_folder),
        "source_label_counts": dict(sorted(source_label_counts.items())),
        "removed_previous_imports": removed_previous_imports,
        "train_counts": dict(train_counts),
        "val_counts": dict(val_counts),
        "total_imported": int(
            sum(count for key, count in train_counts.items() if key != "skipped_existing")
            + sum(count for key, count in val_counts.items() if key != "skipped_existing")
        ),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[INFO] Manifesto da importacao salvo em: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa o MathWriting e rasteriza simbolos compativeis para data/symbols.",
    )
    parser.add_argument(
        "--archive-path",
        default=str(DEFAULT_ARCHIVE_PATH),
        help="Caminho local do arquivo mathwriting-2024.tgz.",
    )
    parser.add_argument(
        "--archive-url",
        default=DEFAULT_ARCHIVE_URL,
        help="URL oficial usada para download do MathWriting.",
    )
    parser.add_argument(
        "--use-excerpt",
        action="store_true",
        help="Usa o excerpt oficial menor para validar o fluxo rapidamente.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Diretorio base do dataset customizado do projeto.",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Caminho do manifesto JSON com o resumo da importacao.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SELECTED_FOLDERS),
        help="Classes a importar. Aceita digits, operators, all, 0-9 e os rotulos + - * / = ( ).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Proporcao dos simbolos enviada para train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed usada na divisao deterministica train/val.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Baixa novamente o arquivo mesmo se ele ja existir em disco.",
    )
    parser.add_argument(
        "--clean-previous-import",
        action="store_true",
        help="Remove apenas arquivos mathwriting_*.png importados anteriormente nas classes selecionadas.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    archive_url = DEFAULT_EXCERPT_URL if args.use_excerpt else args.archive_url
    archive_path = Path(args.archive_path).resolve()
    if args.use_excerpt and archive_path.name == DEFAULT_ARCHIVE_PATH.name:
        archive_path = archive_path.with_name("mathwriting-2024-excerpt.tgz")

    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest_path).resolve()

    selected_folders = expand_requested_folders(args.symbols)
    label_to_folder = build_normalized_label_to_folder_mapping(selected_folders)

    archive_path = download_mathwriting_archive(
        archive_path=archive_path,
        archive_url=archive_url,
        force_download=args.force_download,
    )

    removed_previous_imports = 0
    if args.clean_previous_import:
        removed_previous_imports = remove_previous_imports(output_dir, selected_folders)
        print(f"[INFO] Arquivos antigos do MathWriting removidos: {removed_previous_imports}")

    train_counts, val_counts, source_label_counts = import_mathwriting_symbols(
        archive_path=archive_path,
        output_dir=output_dir,
        label_to_folder=label_to_folder,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    print("[SUCESSO] Importacao MathWriting concluida.")
    print(f"[INFO] Train ratio: {args.train_ratio:.2f}")
    print("[INFO] Quantidade importada no treino:")
    for folder_name in selected_folders:
        print(f"  {folder_name}: {train_counts.get(folder_name, 0)}")
    print("[INFO] Quantidade importada na validacao:")
    for folder_name in selected_folders:
        print(f"  {folder_name}: {val_counts.get(folder_name, 0)}")
    print(f"[INFO] Arquivos ja existentes ignorados no treino: {train_counts['skipped_existing']}")
    print(f"[INFO] Arquivos ja existentes ignorados na validacao: {val_counts['skipped_existing']}")

    write_manifest(
        manifest_path=manifest_path,
        archive_path=archive_path,
        archive_url=archive_url,
        train_ratio=args.train_ratio,
        seed=args.seed,
        selected_folders=selected_folders,
        label_to_folder=label_to_folder,
        train_counts=train_counts,
        val_counts=val_counts,
        source_label_counts=source_label_counts,
        removed_previous_imports=removed_previous_imports,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
