import io
import json
import tarfile
from pathlib import Path

from PIL import Image

from src.data.import_hasyv2 import (
    build_folder_to_latex_mapping,
    build_latex_to_folder_mapping,
    expand_requested_folders,
    import_hasy_split,
    remove_previous_imports,
    write_manifest,
)


def _png_bytes(color: int) -> bytes:
    image = Image.new("L", (32, 32), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_tar_text(archive: tarfile.TarFile, name: str, content: str) -> None:
    data = content.encode("utf-8")
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _write_tar_png(archive: tarfile.TarFile, name: str, color: int) -> None:
    data = _png_bytes(color)
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _create_fake_hasy_archive(archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:bz2") as archive:
        _write_tar_text(
            archive,
            "classification-task/fold-1/train.csv",
            "\n".join(
                [
                    "path,symbol_id,latex,user_id",
                    "../../hasy-data/v2-plus-train.png,196,+,1",
                    "../../hasy-data/v2-minus-train.png,195,-,2",
                    "../../hasy-data/v2-div-train.png,922,/,3",
                    "../../hasy-data/v2-obelus-train.png,526,\\div,4",
                ]
            ),
        )
        _write_tar_text(
            archive,
            "classification-task/fold-1/test.csv",
            "\n".join(
                [
                    "path,symbol_id,latex,user_id",
                    "../../hasy-data/v2-plus-val.png,196,+,5",
                    "../../hasy-data/v2-div-val.png,922,/,6",
                ]
            ),
        )
        _write_tar_png(archive, "hasy-data/v2-plus-train.png", 255)
        _write_tar_png(archive, "hasy-data/v2-minus-train.png", 200)
        _write_tar_png(archive, "hasy-data/v2-div-train.png", 180)
        _write_tar_png(archive, "hasy-data/v2-obelus-train.png", 160)
        _write_tar_png(archive, "hasy-data/v2-plus-val.png", 220)
        _write_tar_png(archive, "hasy-data/v2-div-val.png", 140)


def test_expand_requested_folders_accepts_aliases_and_labels():
    folders = expand_requested_folders(["operators", "+", "div"])

    assert folders == ["plus", "minus", "times", "div"]


def test_import_hasy_split_copies_only_selected_samples(tmp_path):
    archive_path = tmp_path / "HASYv2.tar.bz2"
    output_dir = tmp_path / "symbols"
    _create_fake_hasy_archive(archive_path)

    folder_to_latex = build_folder_to_latex_mapping(["plus", "div"], include_obelus=False)
    latex_to_folder = build_latex_to_folder_mapping(folder_to_latex)

    train_counts = import_hasy_split(
        archive_path=archive_path,
        output_dir=output_dir,
        fold=1,
        split="train",
        latex_to_folder=latex_to_folder,
    )
    val_counts = import_hasy_split(
        archive_path=archive_path,
        output_dir=output_dir,
        fold=1,
        split="val",
        latex_to_folder=latex_to_folder,
    )

    assert train_counts["plus"] == 1
    assert train_counts["div"] == 1
    assert val_counts["plus"] == 1
    assert val_counts["div"] == 1

    assert (output_dir / "train" / "plus" / "hasyv2_fold1_v2-plus-train.png").exists()
    assert (output_dir / "train" / "div" / "hasyv2_fold1_v2-div-train.png").exists()
    assert not (output_dir / "train" / "div" / "hasyv2_fold1_v2-obelus-train.png").exists()


def test_import_hasy_split_can_include_obelus(tmp_path):
    archive_path = tmp_path / "HASYv2.tar.bz2"
    output_dir = tmp_path / "symbols"
    _create_fake_hasy_archive(archive_path)

    folder_to_latex = build_folder_to_latex_mapping(["div"], include_obelus=True)
    latex_to_folder = build_latex_to_folder_mapping(folder_to_latex)

    train_counts = import_hasy_split(
        archive_path=archive_path,
        output_dir=output_dir,
        fold=1,
        split="train",
        latex_to_folder=latex_to_folder,
    )

    assert train_counts["div"] == 2
    assert (output_dir / "train" / "div" / "hasyv2_fold1_v2-obelus-train.png").exists()


def test_remove_previous_imports_and_write_manifest(tmp_path):
    output_dir = tmp_path / "symbols"
    target_file = output_dir / "train" / "plus" / "hasyv2_fold1_v2-plus-train.png"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_bytes(b"png")

    removed = remove_previous_imports(output_dir, ["plus"])
    assert removed == 1
    assert not target_file.exists()

    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        archive_path=tmp_path / "HASYv2.tar.bz2",
        archive_url="https://example.com/HASYv2.tar.bz2",
        fold=1,
        folder_to_latex={"plus": ["+"]},
        train_counts={"plus": 1, "skipped_existing": 0},
        val_counts={"plus": 1, "skipped_existing": 0},
        removed_previous_imports=removed,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "HASYv2"
    assert manifest["selected_folders"] == ["plus"]
    assert manifest["total_imported"] == 2
