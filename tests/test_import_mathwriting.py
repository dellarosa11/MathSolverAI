import io
import json
import tarfile
from pathlib import Path

from PIL import Image

from src.data.import_mathwriting import (
    build_normalized_label_to_folder_mapping,
    build_split_plan,
    expand_requested_folders,
    import_mathwriting_symbols,
    remove_previous_imports,
    write_manifest,
)


def _make_inkml(sample_id: str, label: str, traces: list[str]) -> str:
    trace_lines = "\n".join(
        f'  <trace id="{index}">{trace}</trace>'
        for index, trace in enumerate(traces)
    )
    return f"""<ink xmlns="http://www.w3.org/2003/InkML">
  <annotation type="label">{label}</annotation>
  <annotation type="splitTagOriginal">symbols</annotation>
  <annotation type="sampleId">{sample_id}</annotation>
  <traceFormat>
    <channel name="X" type="decimal" />
    <channel name="Y" type="decimal" />
    <channel name="T" type="decimal" units="ms" />
  </traceFormat>
{trace_lines}
</ink>"""


def _add_text_member(archive: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _create_fake_mathwriting_archive(archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        _add_text_member(archive, "mathwriting-2024-excerpt/archive_readme.md", "dataset")
        _add_text_member(
            archive,
            "mathwriting-2024-excerpt/symbols/a1.inkml",
            _make_inkml("a1", "+", ["0 0 0,10 0 1", "5 -5 0,5 5 1"]),
        )
        _add_text_member(
            archive,
            "mathwriting-2024-excerpt/symbols/a2.inkml",
            _make_inkml("a2", "+", ["0 0 0,8 0 1", "4 -4 0,4 4 1"]),
        )
        _add_text_member(
            archive,
            "mathwriting-2024-excerpt/symbols/b1.inkml",
            _make_inkml("b1", "0", ["0 0 0,8 0 1,8 8 2,0 8 3,0 0 4"]),
        )
        _add_text_member(
            archive,
            "mathwriting-2024-excerpt/symbols/c1.inkml",
            _make_inkml("c1", r"\div", ["0 0 0,10 10 1"]),
        )
        _add_text_member(
            archive,
            "mathwriting-2024-excerpt/symbols/x1.inkml",
            _make_inkml("x1", "x", ["0 0 0,10 10 1", "10 0 0,0 10 1"]),
        )
        _add_text_member(
            archive,
            "mathwriting-2024-excerpt/train/t1.inkml",
            _make_inkml("t1", "+", ["0 0 0,1 1 1"]),
        )


def test_expand_requested_folders_accepts_aliases_and_labels():
    folders = expand_requested_folders(["operators", "0", "+", "=", "x"])

    assert folders[:7] == ["plus", "minus", "times", "div", "equals", "lparen", "rparen"]
    assert "0" in folders
    assert "x" in folders


def test_build_normalized_label_to_folder_mapping_supports_latex_aliases():
    mapping = build_normalized_label_to_folder_mapping(["plus", "div", "equals", "x"])

    assert mapping["+"] == "plus"
    assert mapping[r"\div"] == "div"
    assert mapping["="] == "equals"
    assert mapping["x"] == "x"
    assert mapping["X"] == "x"


def test_build_split_plan_is_deterministic():
    entries = [
        {"folder_name": "plus", "sample_id": "a1"},
        {"folder_name": "plus", "sample_id": "a2"},
        {"folder_name": "plus", "sample_id": "a3"},
        {"folder_name": "plus", "sample_id": "a4"},
    ]

    plan = build_split_plan(entries, train_ratio=0.5, seed=42)

    assert len(plan["train"]) == 2
    assert len(plan["val"]) == 2
    assert plan == build_split_plan(entries, train_ratio=0.5, seed=42)


def test_import_mathwriting_symbols_rasterizes_supported_labels(tmp_path):
    archive_path = tmp_path / "mathwriting.tgz"
    output_dir = tmp_path / "symbols"
    _create_fake_mathwriting_archive(archive_path)

    train_counts, val_counts, source_label_counts = import_mathwriting_symbols(
        archive_path=archive_path,
        output_dir=output_dir,
        label_to_folder=build_normalized_label_to_folder_mapping(["0", "plus", "div", "x"]),
        train_ratio=0.5,
        seed=123,
    )

    assert train_counts["plus"] == 1
    assert val_counts["plus"] == 1
    assert train_counts["0"] == 1
    assert train_counts["div"] == 1
    assert train_counts["x"] == 1
    assert source_label_counts["+"] == 2
    assert source_label_counts["0"] == 1
    assert source_label_counts[r"\div"] == 1
    assert source_label_counts["x"] == 1

    imported_image = Image.open(next((output_dir / "train" / "plus").glob("mathwriting_*.png"))).convert("L")
    assert imported_image.size == (28, 28)
    assert max(imported_image.getdata()) > 0


def test_remove_previous_imports_and_write_manifest(tmp_path):
    output_dir = tmp_path / "symbols"
    target_file = output_dir / "train" / "plus" / "mathwriting_a1.png"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_bytes(b"png")

    removed = remove_previous_imports(output_dir, ["plus"])
    assert removed == 1
    assert not target_file.exists()

    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        archive_path=tmp_path / "mathwriting.tgz",
        archive_url="https://example.com/mathwriting.tgz",
        train_ratio=0.8,
        seed=42,
        selected_folders=["plus"],
        label_to_folder={"+": "plus"},
        train_counts={"plus": 2, "skipped_existing": 0},
        val_counts={"plus": 1, "skipped_existing": 0},
        source_label_counts={"+": 3},
        removed_previous_imports=removed,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "MathWriting"
    assert manifest["selected_folders"] == ["plus"]
    assert manifest["source_label_counts"] == {"+": 3}
