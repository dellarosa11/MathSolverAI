import io
import json
import zipfile
from pathlib import Path

from PIL import Image

from src.data.import_bhmsds import (
    build_source_name_to_folder_mapping,
    build_split_plan,
    expand_requested_folders,
    import_bhmsds_dataset,
    list_source_members,
    remove_previous_imports,
    write_manifest,
)


def _png_bytes(background: int = 255, foreground: int = 0) -> bytes:
    image = Image.new("L", (28, 28), color=background)
    image.putpixel((0, 0), foreground)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _create_fake_bhmsds_archive(archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w") as archive:
        for member_name in [
            "bhmsds-master/README.md",
            "bhmsds-master/symbols/plus-0000.png",
            "bhmsds-master/symbols/plus-0001.png",
            "bhmsds-master/symbols/plus-0002.png",
            "bhmsds-master/symbols/0-0000.png",
            "bhmsds-master/symbols/0-0001.png",
            "bhmsds-master/symbols/slash-0000.png",
            "bhmsds-master/symbols/dot-0000.png",
        ]:
            if member_name.endswith(".png"):
                archive.writestr(member_name, _png_bytes())
            else:
                archive.writestr(member_name, "dataset")


def test_expand_requested_folders_accepts_aliases_and_labels():
    folders = expand_requested_folders(["operators", "0", "+", "digits"])

    assert folders[:3] == ["plus", "minus", "div"]
    assert "0" in folders
    assert "9" in folders


def test_list_source_members_filters_to_supported_sources(tmp_path):
    archive_path = tmp_path / "bhmsds.zip"
    _create_fake_bhmsds_archive(archive_path)

    source_to_folder = build_source_name_to_folder_mapping(["0", "plus", "div"])
    members = list_source_members(archive_path, source_to_folder)

    assert sorted(members) == ["0", "plus", "slash"]
    assert len(members["plus"]) == 3
    assert len(members["0"]) == 2
    assert len(members["slash"]) == 1


def test_build_split_plan_is_deterministic():
    plan = build_split_plan(
        {
            "plus": ["plus-0000", "plus-0001", "plus-0002", "plus-0003", "plus-0004"],
        },
        train_ratio=0.6,
        seed=42,
    )

    assert len(plan["train"]) == 3
    assert len(plan["val"]) == 2
    assert plan == build_split_plan(
        {
            "plus": ["plus-0000", "plus-0001", "plus-0002", "plus-0003", "plus-0004"],
        },
        train_ratio=0.6,
        seed=42,
    )


def test_import_bhmsds_dataset_copies_and_inverts_images(tmp_path):
    archive_path = tmp_path / "bhmsds.zip"
    output_dir = tmp_path / "symbols"
    _create_fake_bhmsds_archive(archive_path)

    train_counts, val_counts = import_bhmsds_dataset(
        archive_path=archive_path,
        output_dir=output_dir,
        source_to_folder=build_source_name_to_folder_mapping(["0", "plus", "div"]),
        train_ratio=0.5,
        seed=123,
    )

    assert train_counts["plus"] == 2
    assert val_counts["plus"] == 1
    assert train_counts["0"] == 1
    assert val_counts["0"] == 1
    assert train_counts["div"] == 1

    imported_image = Image.open(next((output_dir / "train" / "plus").glob("bhmsds_*.png"))).convert("L")
    assert imported_image.getpixel((0, 0)) == 255
    assert imported_image.getpixel((5, 5)) == 0


def test_remove_previous_imports_and_write_manifest(tmp_path):
    output_dir = tmp_path / "symbols"
    target_file = output_dir / "train" / "plus" / "bhmsds_plus_0000.png"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_bytes(b"png")

    removed = remove_previous_imports(output_dir, ["plus"])
    assert removed == 1
    assert not target_file.exists()

    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        archive_path=tmp_path / "bhmsds.zip",
        archive_url="https://example.com/bhmsds.zip",
        source_to_folder={"plus": "plus"},
        train_ratio=0.8,
        seed=42,
        train_counts={"plus": 2, "skipped_existing": 0},
        val_counts={"plus": 1, "skipped_existing": 0},
        removed_previous_imports=removed,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "BHMSDS"
    assert manifest["source_to_folder"] == {"plus": "plus"}
    assert manifest["total_imported"] == 3
