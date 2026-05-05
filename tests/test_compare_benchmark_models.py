from pathlib import Path

from src.utils.compare_benchmark_models import (
    build_comparison_report,
    parse_model_spec,
    rank_model_summary,
    write_comparison_csv,
)


def _make_report(corrected_expression_accuracy, corrected_result_accuracy, sample_overrides=None):
    sample = {
        "sample_id": "s1",
        "image_path": "sample.png",
        "expected_expression": "1+2",
        "expected_result": "3",
        "raw_expression": "1=2",
        "corrected_expression": "1+2",
        "corrected_exact_match": True,
        "corrected_edit_distance": 0,
        "corrected_result": "3",
        "corrected_result_match": True,
        "correction_improved_expression": True,
    }
    if sample_overrides:
        sample.update(sample_overrides)
    return {
        "summary": {
            "sample_count": 1,
            "raw_expression_accuracy": 0.0,
            "corrected_expression_accuracy": corrected_expression_accuracy,
            "corrected_symbol_accuracy": corrected_expression_accuracy,
            "corrected_result_accuracy": corrected_result_accuracy,
        },
        "samples": [sample],
    }


def test_parse_model_spec_supports_alias(tmp_path):
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"weights")

    alias, resolved = parse_model_spec(f"v1={model_path}")

    assert alias == "v1"
    assert resolved == model_path.resolve()


def test_rank_model_summary_prefers_result_accuracy():
    better = rank_model_summary(
        {
            "corrected_result_accuracy": 1.0,
            "corrected_expression_accuracy": 0.8,
            "corrected_symbol_accuracy": 0.8,
            "raw_expression_accuracy": 0.7,
        }
    )
    worse = rank_model_summary(
        {
            "corrected_result_accuracy": 0.5,
            "corrected_expression_accuracy": 1.0,
            "corrected_symbol_accuracy": 1.0,
            "raw_expression_accuracy": 1.0,
        }
    )

    assert better > worse


def test_build_comparison_report_ranks_models_and_counts_wins(tmp_path):
    model_paths = {
        "v1": Path(tmp_path / "v1.pth"),
        "v2": Path(tmp_path / "v2.pth"),
    }
    report = build_comparison_report(
        "benchmarks/local_photos.json",
        model_reports={
            "v1": _make_report(1.0, 1.0),
            "v2": _make_report(
                0.0,
                0.0,
                sample_overrides={
                    "corrected_expression": "12",
                    "corrected_exact_match": False,
                    "corrected_edit_distance": 2,
                    "corrected_result": "12",
                    "corrected_result_match": False,
                    "correction_improved_expression": False,
                },
            ),
        },
        model_paths=model_paths,
    )

    assert report["ranking"][0]["model"] == "v1"
    assert report["expression_win_counts"]["v1"] == 1
    assert report["result_win_counts"]["v1"] == 1
    assert report["samples"][0]["expression_winners"] == ["v1"]


def test_write_comparison_csv_exports_model_columns(tmp_path):
    report = build_comparison_report(
        "benchmarks/local_photos.json",
        model_reports={
            "v1": _make_report(1.0, 1.0),
            "v2": _make_report(
                0.0,
                0.0,
                sample_overrides={
                    "corrected_expression": "12",
                    "corrected_exact_match": False,
                    "corrected_edit_distance": 2,
                    "corrected_result": "12",
                    "corrected_result_match": False,
                    "correction_improved_expression": False,
                },
            ),
        },
        model_paths={
            "v1": Path(tmp_path / "v1.pth"),
            "v2": Path(tmp_path / "v2.pth"),
        },
    )
    csv_path = tmp_path / "comparison.csv"

    write_comparison_csv(report, csv_path)

    content = csv_path.read_text(encoding="utf-8")
    assert "v1_corrected_expression" in content
    assert "v2_corrected_result_match" in content
