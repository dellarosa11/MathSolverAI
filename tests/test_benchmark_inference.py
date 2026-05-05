import json
from pathlib import Path
from types import SimpleNamespace

from src.utils.benchmark_inference import (
    BenchmarkSample,
    evaluate_samples,
    levenshtein_distance,
    load_benchmark_samples,
    write_benchmark_csv,
)


class DummyApp:
    def __init__(self):
        self.solver = SimpleNamespace(
            normalize_expression=lambda expression: expression,
            solve=lambda expression: 3 if expression in {"1+2", "1+2=3"} else expression,
        )

    def recognize_expression(self, image_path, top_k=3):
        return SimpleNamespace(expression="1=2", symbols=[])

    def improve_expression(self, recognition, **kwargs):
        candidate = SimpleNamespace(
            expression="1+2",
            normalized_expression="1+2",
            score=1.2,
            valid=True,
            solvable=True,
        )
        return SimpleNamespace(
            raw_expression=recognition.expression,
            corrected_expression="1+2",
            changed=True,
            selected_candidate=candidate,
            candidates=[candidate],
        )


def test_levenshtein_distance_counts_edits():
    assert levenshtein_distance("123", "13") == 1
    assert levenshtein_distance("1+2", "1+2") == 0


def test_load_benchmark_samples_accepts_json_list(tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"img")
    manifest_path = tmp_path / "samples.json"
    manifest_path.write_text(
        json.dumps([{"image": "sample.png", "expected_expression": "1+2", "expected_result": "3"}]),
        encoding="utf-8",
    )

    samples = load_benchmark_samples(manifest_path)

    assert samples == [
        BenchmarkSample(
            image_path=image_path.resolve(),
            expected_expression="1+2",
            expected_result="3",
            sample_id="sample_0001",
        )
    ]


def test_evaluate_samples_tracks_raw_and_corrected_metrics(tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"img")
    samples = [BenchmarkSample(image_path=image_path, expected_expression="1+2", expected_result="3", sample_id="s1")]

    report = evaluate_samples(DummyApp(), samples)

    assert report["summary"]["sample_count"] == 1
    assert report["summary"]["raw_expression_accuracy"] == 0.0
    assert report["summary"]["corrected_expression_accuracy"] == 1.0
    assert report["summary"]["corrected_result_accuracy"] == 1.0
    assert report["summary"]["samples_improved_by_correction"] == 1
    assert report["summary"]["result_samples_improved_by_correction"] == 1


def test_write_benchmark_csv_exports_rows(tmp_path):
    report = {
        "summary": {},
        "samples": [
            {
                "sample_id": "s1",
                "image_path": "sample.png",
                "expected_expression": "1+2",
                "raw_expression": "1=2",
                "corrected_expression": "1+2",
                "raw_exact_match": False,
                "corrected_exact_match": True,
                "raw_edit_distance": 1,
                "corrected_edit_distance": 0,
                "expected_result": "3",
                "raw_result": None,
                "corrected_result": "3",
                "raw_result_match": False,
                "corrected_result_match": True,
                "correction_changed_expression": True,
                "correction_improved_expression": True,
                "correction_improved_result": True,
            }
        ],
    }
    csv_path = tmp_path / "report.csv"

    write_benchmark_csv(report, csv_path)

    content = csv_path.read_text(encoding="utf-8")
    assert "sample_id" in content
    assert "corrected_expression" in content
    assert "1+2" in content
