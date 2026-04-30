from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data.class_config import DIGIT_CLASSES, get_folder_name_for_label


DEFAULT_ARCHIVE_URL = "https://codeload.github.com/wblachowski/bhmsds/zip/refs/heads/master"
DEFAULT_ARCHIVE_PATH = BASE_DIR / "data" / "external" / "bhmsds" / "bhmsds.zip"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "symbols"
DEFAULT_MANIFEST_PATH = BASE_DIR / "data" / "external" / "bhmsds" / "import_manifest.json"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_SEED = 42

SOURCE_NAME_TO_FOLDER: Dict[str, str] = {
    **{digit: digit for digit in DIGIT_CLASSES},
    "plus": "plus",
    "minus": "minus",
    "slash": "div",
}
DEFAULT_SELECTED_FOLDERS = tuple(DIGIT_CLASSES + ["plus", "minus", "div"])
OPERATORS_ONLY = ("plus", "minus", "div")


def download_bhmsds_archive(
    archive_path: Path,
    archive_url: str = DEFAULT_ARCHIVE_URL,
    force_download: bool = False,
) -> Path:
    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if archive_path.exists() and not force_download:
        print(f"[INFO] Reutilizando arquivo ja baixado: {archive_path}")
        return archive_path

    print(f"[INFO] Baixando BHMSDS de: {archive_url}")
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
                f"Classe BHMSDS nao suportada: {item}. Use digits, operators, all ou uma destas classes: {allowed}."
            ) from exc

        if folder_name not in DEFAULT_SELECTED_FOLDERS:
            allowed = ", ".join(DEFAULT_SELECTED_FOLDERS)
            raise ValueError(
                f"O BHMSDS nao cobre a classe '{item}' neste importador. "
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


def build_source_name_to_folder_mapping(selected_folders: Sequence[str]) -> dict[str, str]:
    folder_set = set(selected_folders)
    return {
        source_name: folder_name
        for source_name, folder_name in SOURCE_NAME_TO_FOLDER.items()
        if folder_name in folder_set
    }


def list_source_members(
    archive_path: Path,
    source_to_folder: Mapping[str, str],
) -> dict[str, list[str]]:
    members_by_source = {source_name: [] for source_name in source_to_folder}

    with zipfile.ZipFile(archive_path) as archive:
        for member_name in archive.namelist():
            if not member_name.endswith(".png") or "/symbols/" not in member_name:
                continue

            source_name = Path(member_name).name.split("-", 1)[0]
            if source_name in members_by_source:
                members_by_source[source_name].append(member_name)

    return members_by_source


def build_split_plan(
    members_by_source: Mapping[str, Sequence[str]],
    *,
    train_ratio: float,
    seed: int,
) -> dict[str, list[tuple[str, str, str]]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio deve ficar entre 0 e 1.")

    plan = {"train": [], "val": []}
    for source_name, member_names in members_by_source.items():
        shuffled = list(member_names)
        random.Random(f"{seed}:{source_name}").shuffle(shuffled)

        if len(shuffled) <= 1:
            train_count = len(shuffled)
        else:
            train_count = int(round(len(shuffled) * train_ratio))
            train_count = max(1, min(len(shuffled) - 1, train_count))

        train_members = shuffled[:train_count]
        val_members = shuffled[train_count:]

        plan["train"].extend((source_name, "train", member_name) for member_name in train_members)
        plan["val"].extend((source_name, "val", member_name) for member_name in val_members)

    return plan


def _convert_bhmsds_image(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        grayscale = image.convert("L")
        inverted = ImageOps.invert(grayscale)
        buffer = io.BytesIO()
        inverted.save(buffer, format="PNG")
        return buffer.getvalue()


def remove_previous_imports(output_dir: Path, selected_folders: Sequence[str]) -> int:
    removed = 0
    for split in ("train", "val"):
        for folder_name in selected_folders:
            target_dir = output_dir / split / folder_name
            if not target_dir.exists():
                continue
            for image_path in target_dir.glob("bhmsds_*.png"):
                image_path.unlink()
                removed += 1
    return removed


def import_bhmsds_dataset(
    archive_path: Path,
    output_dir: Path,
    *,
    source_to_folder: Mapping[str, str],
    train_ratio: float,
    seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    members_by_source = list_source_members(archive_path, source_to_folder)
    split_plan = build_split_plan(members_by_source, train_ratio=train_ratio, seed=seed)

    train_counts = {folder_name: 0 for folder_name in sorted(set(source_to_folder.values()))}
    val_counts = {folder_name: 0 for folder_name in sorted(set(source_to_folder.values()))}
    train_counts["skipped_existing"] = 0
    val_counts["skipped_existing"] = 0

    with zipfile.ZipFile(archive_path) as archive:
        for source_name, split_name, member_name in split_plan["train"] + split_plan["val"]:
            folder_name = source_to_folder[source_name]
            destination_dir = output_dir / split_name / folder_name
            destination_dir.mkdir(parents=True, exist_ok=True)

            source_stem = Path(member_name).stem.replace("-", "_")
            destination_path = destination_dir / f"bhmsds_{source_stem}.png"
            split_counts = train_counts if split_name == "train" else val_counts

            if destination_path.exists():
                split_counts["skipped_existing"] += 1
                continue

            converted_bytes = _convert_bhmsds_image(archive.read(member_name))
            destination_path.write_bytes(converted_bytes)
            split_counts[folder_name] += 1

    return train_counts, val_counts


def write_manifest(
    manifest_path: Path,
    *,
    archive_path: Path,
    archive_url: str,
    source_to_folder: Mapping[str, str],
    train_ratio: float,
    seed: int,
    train_counts: Mapping[str, int],
    val_counts: Mapping[str, int],
    removed_previous_imports: int,
) -> None:
    manifest: dict[str, Any] = {
        "dataset": "BHMSDS",
        "archive_path": str(archive_path),
        "archive_url": archive_url,
        "source_to_folder": dict(source_to_folder),
        "train_ratio": train_ratio,
        "seed": seed,
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
        description="Baixa o BHMSDS e importa digitos e operadores compativeis para data/symbols.",
    )
    parser.add_argument(
        "--archive-path",
        default=str(DEFAULT_ARCHIVE_PATH),
        help="Caminho local do arquivo bhmsds.zip.",
    )
    parser.add_argument(
        "--archive-url",
        default=DEFAULT_ARCHIVE_URL,
        help="URL usada para download do BHMSDS.",
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
        help="Classes a importar. Aceita digits, operators, all, 0-9, plus, minus, div ou os rotulos + - /.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Proporcao das imagens enviada para train.",
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
        help="Remove apenas arquivos bhmsds_*.png importados anteriormente nas classes selecionadas.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    archive_path = Path(args.archive_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest_path).resolve()

    selected_folders = expand_requested_folders(args.symbols)
    source_to_folder = build_source_name_to_folder_mapping(selected_folders)

    archive_path = download_bhmsds_archive(
        archive_path=archive_path,
        archive_url=args.archive_url,
        force_download=args.force_download,
    )

    removed_previous_imports = 0
    if args.clean_previous_import:
        removed_previous_imports = remove_previous_imports(output_dir, selected_folders)
        print(f"[INFO] Arquivos antigos do BHMSDS removidos: {removed_previous_imports}")

    train_counts, val_counts = import_bhmsds_dataset(
        archive_path=archive_path,
        output_dir=output_dir,
        source_to_folder=source_to_folder,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    print("[SUCESSO] Importacao BHMSDS concluida.")
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
        archive_url=args.archive_url,
        source_to_folder=source_to_folder,
        train_ratio=args.train_ratio,
        seed=args.seed,
        train_counts=train_counts,
        val_counts=val_counts,
        removed_previous_imports=removed_previous_imports,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
