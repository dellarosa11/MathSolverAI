from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_training_report(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _rank_worst_classes(per_class_accuracy: dict[str, float | None], limit: int) -> list[tuple[str, float]]:
    ranked = [
        (class_name, float(accuracy))
        for class_name, accuracy in per_class_accuracy.items()
        if accuracy is not None
    ]
    ranked.sort(key=lambda item: item[1])
    return ranked[:limit]


def print_report_summary(report: dict[str, Any], limit: int = 5) -> None:
    final_validation = report.get("final_validation", {})
    per_class_accuracy = final_validation.get("per_class_accuracy", {})
    confused_pairs = final_validation.get("most_confused_pairs", [])

    print("[INFO] Resumo do treino")
    print(f"  Arquitetura: {report.get('architecture')}")
    print(f"  Melhor epoca: {report.get('best_epoch')}")
    print(f"  Melhor val_acc: {report.get('best_val_accuracy', 0.0) * 100:.2f}%")
    print(f"  Early stopping: {'sim' if report.get('stopped_early') else 'nao'}")

    print("[INFO] Piores classes:")
    for class_name, accuracy in _rank_worst_classes(per_class_accuracy, limit=limit):
        print(f"  {class_name}: {accuracy * 100:.2f}%")

    if confused_pairs:
        print("[INFO] Maiores confusoes:")
        for pair in confused_pairs[:limit]:
            print(
                f"  {pair['true_label']} -> {pair['predicted_label']}: "
                f"{pair['count']} ocorrencias"
            )
    else:
        print("[INFO] Nenhuma confusao registrada no relatorio.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume um relatorio JSON de treino do MathSolverAI.",
    )
    parser.add_argument(
        "report_path",
        help="Caminho do relatorio JSON gerado pelo treino.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Quantidade maxima de classes/confusoes exibidas.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    report = load_training_report(args.report_path)
    print_report_summary(report, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
