from src.utils.analyze_training_report import load_training_report


def test_load_training_report_reads_json(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text('{"architecture": "cnn", "best_epoch": 2}', encoding="utf-8")

    report = load_training_report(report_path)

    assert report["architecture"] == "cnn"
    assert report["best_epoch"] == 2
