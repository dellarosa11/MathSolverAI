import json

from src.utils.import_debug_corrections import import_corrections, resolve_target_labels


def test_resolve_target_labels_from_expected_expression():
    summary = {"symbols": [{"index": 1}, {"index": 2}, {"index": 3}]}

    labels = resolve_target_labels(summary, expected_expression="1+2")

    assert labels == ["1", "+", "2"]


def test_import_corrections_copies_symbol_crops(tmp_path):
    summary_dir = tmp_path / "debug"
    summary_dir.mkdir()
    crop_path = summary_dir / "01_one_nn.png"
    crop_path.write_bytes(b"fake")
    summary = {
        "image_path": "sample.png",
        "symbols": [
            {
                "index": 1,
                "label": "7",
                "confidence": 0.9,
                "nn_crop_file": "01_one_nn.png",
                "raw_crop_file": "01_one_raw.png",
            }
        ],
    }
    summary_path = summary_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = import_corrections(
        summary,
        summary_path=summary_path,
        labels=["1"],
        output_dir=tmp_path / "symbols",
        split="train",
    )

    assert report["imported_count"] == 1
    destination = report["files"][0]["destination"]
    assert destination.endswith("train\\1\\sample_s01_1.png") or destination.endswith("train/1/sample_s01_1.png")
