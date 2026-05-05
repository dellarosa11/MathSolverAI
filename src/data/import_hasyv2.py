from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data.class_config import get_folder_name_for_label


DEFAULT_ARCHIVE_URL = "https://zenodo.org/records/259444/files/HASYv2.tar.bz2?download=1"
DEFAULT_ARCHIVE_PATH = BASE_DIR / "data" / "external" / "hasyv2" / "HASYv2.tar.bz2"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "symbols"
DEFAULT_MANIFEST_PATH = BASE_DIR / "data" / "external" / "hasyv2" / "import_manifest.json"

AVAILABLE_FOLDER_TO_LATEX: Dict[str, list[str]] = {
    "plus": ["+"],
    "minus": ["-"],
    "times": ["\\times"],
    "div": ["/"],
}
OPTIONAL_FOLDER_TO_LATEX: Dict[str, list[str]] = {
    "div": ["\\div"],
}
DEFAULT_SELECTED_FOLDERS = tuple(AVAILABLE_FOLDER_TO_LATEX.keys())


@dataclass(frozen=True)
class HasyRecord:
    archive_path: str
    symbol_id: str
    latex: str
    user_id: str


def download_hasy_archive(
    archive_path: Path,
    archive_url: str = DEFAULT_ARCHIVE_URL,
    force_download: bool = False,
) -> Path:
    """Baixa o arquivo oficial do HASYv2 quando ele ainda nao existir localmente."""
    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if archive_path.exists() and not force_download:
        print(f"[INFO] Reutilizando arquivo ja baixado: {archive_path}")
        return archive_path

    print(f"[INFO] Baixando HASYv2 de: {archive_url}")
    with urllib.request.urlopen(archive_url) as response, archive_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)

    print(f"[SUCESSO] Arquivo salvo em: {archive_path}")
    return archive_path


def expand_requested_folders(requested: Sequence[str]) -> list[str]:
    """
    Converte nomes amigaveis e rotulos canonicos para as pastas suportadas pelo importador.
    """
    if not requested:
        return list(DEFAULT_SELECTED_FOLDERS)

    expanded: list[str] = []
    for item in requested:
        normalized = item.strip().lower()
        if not normalized:
            continue
        if normalized in {"all", "operators"}:
            expanded.extend(DEFAULT_SELECTED_FOLDERS)
            continue
        if normalized in AVAILABLE_FOLDER_TO_LATEX:
            expanded.append(normalized)
            continue

        try:
            folder_name = get_folder_name_for_label(item)
        except KeyError as exc:
            allowed = ", ".join(DEFAULT_SELECTED_FOLDERS)
            raise ValueError(
                f"Classe HASYv2 nao suportada: {item}. Use uma destas opcoes: {allowed}, operators ou all."
            ) from exc

        if folder_name not in AVAILABLE_FOLDER_TO_LATEX:
            allowed = ", ".join(DEFAULT_SELECTED_FOLDERS)
            raise ValueError(
                f"O HASYv2 nao cobre a classe '{item}' neste importador. "
                f"Classes disponiveis: {allowed}."
            )
        expanded.append(folder_name)

    deduplicated: list[str] = []
    seen = set()
    for folder_name in expanded:
        if folder_name not in seen:
            deduplicated.append(folder_name)
            seen.add(folder_name)
    return deduplicated


def build_folder_to_latex_mapping(
    selected_folders: Sequence[str],
    include_obelus: bool = False,
) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {
        folder_name: list(AVAILABLE_FOLDER_TO_LATEX[folder_name])
        for folder_name in selected_folders
    }
    if include_obelus and "div" in mapping:
        mapping["div"].extend(OPTIONAL_FOLDER_TO_LATEX["div"])
    return mapping


def build_latex_to_folder_mapping(folder_to_latex: Mapping[str, Sequence[str]]) -> dict[str, str]:
    latex_to_folder: dict[str, str] = {}
    for folder_name, latex_values in folder_to_latex.items():
        for latex_value in latex_values:
            latex_to_folder[latex_value] = folder_name
    return latex_to_folder


def normalize_archive_member_path(raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/")
    marker = "hasy-data/"
    marker_index = normalized.find(marker)
    if marker_index >= 0:
        return normalized[marker_index:]
    return normalized.lstrip("./")


def load_hasy_records(
    archive_path: Path,
    fold: int,
    split: str,
) -> list[HasyRecord]:
    with tarfile.open(archive_path, "r:bz2") as archive:
        return load_hasy_records_from_archive(archive, fold=fold, split=split)


def load_hasy_records_from_archive(
    archive: tarfile.TarFile,
    fold: int,
    split: str,
) -> list[HasyRecord]:
    if fold < 1 or fold > 10:
        raise ValueError("O fold do HASYv2 deve ficar entre 1 e 10.")

    csv_name = "train.csv" if split == "train" else "test.csv"
    member_name = f"classification-task/fold-{fold}/{csv_name}"

    with archive.extractfile(member_name) as csv_file:
        if csv_file is None:
            raise FileNotFoundError(f"Arquivo nao encontrado dentro do tar: {member_name}")
        reader = csv.DictReader(TextIOWrapper(csv_file, encoding="utf-8"))
        return [
            HasyRecord(
                archive_path=normalize_archive_member_path(row["path"]),
                symbol_id=row["symbol_id"],
                latex=row["latex"],
                user_id=row["user_id"],
            )
            for row in reader
        ]


def remove_previous_imports(output_dir: Path, selected_folders: Sequence[str]) -> int:
    removed = 0
    for split in ("train", "val"):
        for folder_name in selected_folders:
            target_dir = output_dir / split / folder_name
            if not target_dir.exists():
                continue
            for image_path in target_dir.glob("hasyv2_*.png"):
                image_path.unlink()
                removed += 1
    return removed


def import_hasy_split(
    archive_path: Path,
    output_dir: Path,
    fold: int,
    split: str,
    latex_to_folder: Mapping[str, str],
) -> dict[str, int]:
    split_name = "train" if split == "train" else "val"
    counts = {folder_name: 0 for folder_name in set(latex_to_folder.values())}
    skipped_existing = 0

    with tarfile.open(archive_path, "r:bz2") as archive:
        records = load_hasy_records_from_archive(archive, fold=fold, split=split)
        member_lookup = {
            member.name: member
            for member in archive.getmembers()
            if member.isfile()
        }
        for record in records:
            folder_name = latex_to_folder.get(record.latex)
            if folder_name is None:
                continue

            destination_dir = output_dir / split_name / folder_name
            destination_dir.mkdir(parents=True, exist_ok=True)

            source_stem = Path(record.archive_path).stem
            destination_path = destination_dir / f"hasyv2_fold{fold}_{source_stem}.png"
            if destination_path.exists():
                skipped_existing += 1
                continue

            member = member_lookup.get(record.archive_path)
            if member is None:
                raise FileNotFoundError(
                    f"Imagem nao encontrada dentro do tar: {record.archive_path}"
                )
            source_file = archive.extractfile(member)
            if source_file is None:
                raise FileNotFoundError(
                    f"Nao foi possivel extrair a imagem do tar: {record.archive_path}"
                )

            with source_file, destination_path.open("wb") as output_file:
                shutil.copyfileobj(source_file, output_file)
            counts[folder_name] += 1

    counts["skipped_existing"] = skipped_existing
    return counts


def import_hasy_dataset(
    archive_path: Path,
    output_dir: Path,
    fold: int,
    latex_to_folder: Mapping[str, str],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Importa treino e validacao em uma unica passada sequencial pelo tar.bz2.
    """
    folder_names = sorted(set(latex_to_folder.values()))
    train_counts = {folder_name: 0 for folder_name in folder_names}
    val_counts = {folder_name: 0 for folder_name in folder_names}
    train_counts["skipped_existing"] = 0
    val_counts["skipped_existing"] = 0

    train_records = load_hasy_records(archive_path, fold=fold, split="train")
    val_records = load_hasy_records(archive_path, fold=fold, split="val")

    destination_plan: dict[str, tuple[str, str, Path]] = {}
    for split_name, records in (("train", train_records), ("val", val_records)):
        for record in records:
            folder_name = latex_to_folder.get(record.latex)
            if folder_name is None:
                continue

            destination_dir = output_dir / split_name / folder_name
            destination_dir.mkdir(parents=True, exist_ok=True)

            source_stem = Path(record.archive_path).stem
            destination_path = destination_dir / f"hasyv2_fold{fold}_{source_stem}.png"
            destination_plan[record.archive_path] = (split_name, folder_name, destination_path)

    remaining_paths = set(destination_plan.keys())
    with tarfile.open(archive_path, "r:bz2") as archive:
        for member in archive:
            if not member.isfile():
                continue

            planned_target = destination_plan.get(member.name)
            if planned_target is None:
                continue

            split_name, folder_name, destination_path = planned_target
            split_counts = train_counts if split_name == "train" else val_counts
            if destination_path.exists():
                split_counts["skipped_existing"] += 1
                remaining_paths.discard(member.name)
                continue

            source_file = archive.extractfile(member)
            if source_file is None:
                raise FileNotFoundError(
                    f"Nao foi possivel extrair a imagem do tar: {member.name}"
                )

            with source_file, destination_path.open("wb") as output_file:
                shutil.copyfileobj(source_file, output_file)
            split_counts[folder_name] += 1
            remaining_paths.discard(member.name)

            if not remaining_paths:
                break

    if remaining_paths:
        missing_paths = ", ".join(sorted(list(remaining_paths))[:5])
        raise FileNotFoundError(
            f"Algumas imagens planejadas nao foram encontradas no arquivo HASYv2. Exemplos: {missing_paths}"
        )

    return train_counts, val_counts


def write_manifest(
    manifest_path: Path,
    *,
    archive_path: Path,
    archive_url: str,
    fold: int,
    folder_to_latex: Mapping[str, Sequence[str]],
    train_counts: Mapping[str, int],
    val_counts: Mapping[str, int],
    removed_previous_imports: int,
) -> None:
    manifest = {
        "dataset": "HASYv2",
        "archive_path": str(archive_path),
        "archive_url": archive_url,
        "fold": fold,
        "selected_folders": list(folder_to_latex.keys()),
        "source_latex_by_folder": {
            folder_name: list(latex_values)
            for folder_name, latex_values in folder_to_latex.items()
        },
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
        description="Baixa o HASYv2 oficial e importa simbolos selecionados para data/symbols.",
    )
    parser.add_argument(
        "--archive-path",
        default=str(DEFAULT_ARCHIVE_PATH),
        help="Caminho local do arquivo HASYv2.tar.bz2.",
    )
    parser.add_argument(
        "--archive-url",
        default=DEFAULT_ARCHIVE_URL,
        help="URL oficial usada para download do HASYv2.",
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
        "--fold",
        type=int,
        default=1,
        help="Fold oficial do HASYv2 usado para train/val.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SELECTED_FOLDERS),
        help="Classes a importar. Aceita plus minus times div, operators, all ou os rotulos + - * /.",
    )
    parser.add_argument(
        "--include-obelus",
        action="store_true",
        help="Tambem importa o simbolo \\div para a classe div.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Baixa novamente o arquivo mesmo se ele ja existir em disco.",
    )
    parser.add_argument(
        "--clean-previous-import",
        action="store_true",
        help="Remove apenas arquivos hasyv2_*.png importados anteriormente nas classes selecionadas.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    archive_path = Path(args.archive_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest_path).resolve()

    selected_folders = expand_requested_folders(args.symbols)
    folder_to_latex = build_folder_to_latex_mapping(
        selected_folders,
        include_obelus=args.include_obelus,
    )
    latex_to_folder = build_latex_to_folder_mapping(folder_to_latex)

    archive_path = download_hasy_archive(
        archive_path=archive_path,
        archive_url=args.archive_url,
        force_download=args.force_download,
    )

    removed_previous_imports = 0
    if args.clean_previous_import:
        removed_previous_imports = remove_previous_imports(output_dir, selected_folders)
        print(f"[INFO] Arquivos antigos do HASY removidos: {removed_previous_imports}")

    train_counts, val_counts = import_hasy_dataset(
        archive_path=archive_path,
        output_dir=output_dir,
        fold=args.fold,
        latex_to_folder=latex_to_folder,
    )

    print("[SUCESSO] Importacao HASYv2 concluida.")
    print(f"[INFO] Fold utilizado: {args.fold}")
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
        fold=args.fold,
        folder_to_latex=folder_to_latex,
        train_counts=train_counts,
        val_counts=val_counts,
        removed_previous_imports=removed_previous_imports,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
