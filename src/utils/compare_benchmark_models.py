from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from main import MathSolverAI
from src.utils.benchmark_inference import evaluate_samples, load_benchmark_samples


def parse_model_spec(spec: str) -> tuple[str, Path]:
    raw_spec = spec.strip()
    if not raw_spec:
        raise ValueError("Especificacao de modelo vazia.")

    if "=" in raw_spec:
        alias, path_text = raw_spec.split("=", 1)
        alias = alias.strip()
        path = Path(path_text.strip())
    else:
        path = Path(raw_spec)
        alias = path.stem

    if not alias:
        raise ValueError(f"Alias invalido na especificacao de modelo: {spec}")
    if not path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {path}")

    return alias, path.resolve()


def rank_model_summary(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    corrected_result = summary.get("corrected_result_accuracy")
    return (
        -1.0 if corrected_result is None else float(corrected_result),
        float(summary["corrected_expression_accuracy"]),
        float(summary["corrected_symbol_accuracy"]),
        float(summary["raw_expression_accuracy"]),
    )


def _sample_score(model_sample: dict[str, Any]) -> tuple[int, int, int, float]:
    result_match = model_sample.get("corrected_result_match")
    return (
        int(bool(result_match)) if result_match is not None else -1,
        int(bool(model_sample.get("corrected_exact_match"))),
        -int(model_sample.get("corrected_edit_distance", 999999)),
        1.0 if model_sample.get("correction_improved_expression") else 0.0,
    )


def build_comparison_report(
    manifest_path: str | Path,
    model_reports: dict[str, dict[str, Any]],
    model_paths: dict[str, Path],
) -> dict[str, Any]:
    if not model_reports:
        raise ValueError("Nenhum relatorio de modelo foi informado para comparacao.")

    first_alias = next(iter(model_reports))
    sample_count = model_reports[first_alias]["summary"]["sample_count"]

    for alias, report in model_reports.items():
        if report["summary"]["sample_count"] != sample_count:
            raise ValueError(
                "Todos os modelos precisam ter sido executados sobre o mesmo numero de amostras. "
                f"{alias} tem {report['summary']['sample_count']} amostras, esperado {sample_count}."
            )

    ranking = [
        {
            "model": alias,
            "model_path": str(model_paths[alias]),
            "summary": report["summary"],
        }
        for alias, report in model_reports.items()
    ]
    ranking.sort(key=lambda item: rank_model_summary(item["summary"]), reverse=True)

    sample_comparisons: list[dict[str, Any]] = []
    expression_win_counts = {alias: 0 for alias in model_reports}
    result_win_counts = {alias: 0 for alias in model_reports}
    expression_unique_wins = {alias: 0 for alias in model_reports}
    result_unique_wins = {alias: 0 for alias in model_reports}
    expression_ties = 0
    result_ties = 0

    for sample_index in range(sample_count):
        by_model: dict[str, dict[str, Any]] = {
            alias: report["samples"][sample_index]
            for alias, report in model_reports.items()
        }
        first_sample = next(iter(by_model.values()))
        sample_scores = {alias: _sample_score(sample) for alias, sample in by_model.items()}
        best_score = max(sample_scores.values())
        expression_winners = [
            alias
            for alias, score in sample_scores.items()
            if score == best_score
        ]
        for alias in expression_winners:
            expression_win_counts[alias] += 1
        if len(expression_winners) == 1:
            expression_unique_wins[expression_winners[0]] += 1
        else:
            expression_ties += 1

        result_scores = {
            alias: (
                int(bool(sample.get("corrected_result_match")))
                if sample.get("corrected_result_match") is not None
                else -1
            )
            for alias, sample in by_model.items()
        }
        best_result_score = max(result_scores.values())
        result_winners = [
            alias
            for alias, score in result_scores.items()
            if score == best_result_score and score >= 0
        ]
        for alias in result_winners:
            result_win_counts[alias] += 1
        if not result_winners:
            pass
        elif len(result_winners) == 1:
            result_unique_wins[result_winners[0]] += 1
        else:
            result_ties += 1

        sample_comparisons.append(
            {
                "sample_id": first_sample["sample_id"],
                "image_path": first_sample["image_path"],
                "expected_expression": first_sample["expected_expression"],
                "expected_result": first_sample.get("expected_result"),
                "expression_winners": expression_winners,
                "result_winners": result_winners,
                "models": by_model,
            }
        )

    return {
        "manifest_path": str(Path(manifest_path).resolve()),
        "ranking": ranking,
        "expression_win_counts": expression_win_counts,
        "expression_unique_wins": expression_unique_wins,
        "expression_ties": expression_ties,
        "result_win_counts": result_win_counts,
        "result_unique_wins": result_unique_wins,
        "result_ties": result_ties,
        "samples": sample_comparisons,
    }


def print_comparison_summary(report: dict[str, Any]) -> None:
    print("[INFO] Comparacao de modelos")
    print(f"  Manifesto: {report['manifest_path']}")
    print(f"  Amostras: {len(report['samples'])}")

    for index, item in enumerate(report["ranking"], start=1):
        summary = item["summary"]
        result_accuracy = summary.get("corrected_result_accuracy")
        result_text = "n/d" if result_accuracy is None else f"{result_accuracy * 100:.2f}%"
        print(
            f"  {index}. {item['model']} | "
            f"expr corrigida={summary['corrected_expression_accuracy'] * 100:.2f}% | "
            f"resultado corrigido={result_text} | "
            f"simbolo corrigido={summary['corrected_symbol_accuracy'] * 100:.2f}%"
        )

    winner = report["ranking"][0]
    print(f"[INFO] Melhor modelo geral: {winner['model']}")

    print("[INFO] Vitorias por amostra (expressao)")
    for alias, wins in sorted(report["expression_win_counts"].items(), key=lambda item: item[1], reverse=True):
        print(f"  {alias}: {wins}")
    print(f"  empates: {report['expression_ties']}")
    print("[INFO] Vitorias unicas por amostra (expressao)")
    for alias, wins in sorted(report["expression_unique_wins"].items(), key=lambda item: item[1], reverse=True):
        print(f"  {alias}: {wins}")

    if any(report["result_win_counts"].values()):
        print("[INFO] Vitorias por amostra (resultado)")
        for alias, wins in sorted(report["result_win_counts"].items(), key=lambda item: item[1], reverse=True):
            print(f"  {alias}: {wins}")
        print(f"  empates: {report['result_ties']}")
        print("[INFO] Vitorias unicas por amostra (resultado)")
        for alias, wins in sorted(report["result_unique_wins"].items(), key=lambda item: item[1], reverse=True):
            print(f"  {alias}: {wins}")


def write_comparison_csv(report: dict[str, Any], csv_path: str | Path) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_names = [item["model"] for item in report["ranking"]]
    fieldnames = [
        "sample_id",
        "image_path",
        "expected_expression",
        "expected_result",
        "expression_winners",
        "result_winners",
    ]
    for alias in model_names:
        fieldnames.extend(
            [
                f"{alias}_raw_expression",
                f"{alias}_corrected_expression",
                f"{alias}_corrected_exact_match",
                f"{alias}_corrected_edit_distance",
                f"{alias}_corrected_result",
                f"{alias}_corrected_result_match",
            ]
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in report["samples"]:
            row = {
                "sample_id": sample["sample_id"],
                "image_path": sample["image_path"],
                "expected_expression": sample["expected_expression"],
                "expected_result": sample["expected_result"],
                "expression_winners": ",".join(sample["expression_winners"]),
                "result_winners": ",".join(sample["result_winners"]),
            }
            for alias in model_names:
                model_sample = sample["models"][alias]
                row.update(
                    {
                        f"{alias}_raw_expression": model_sample["raw_expression"],
                        f"{alias}_corrected_expression": model_sample["corrected_expression"],
                        f"{alias}_corrected_exact_match": model_sample["corrected_exact_match"],
                        f"{alias}_corrected_edit_distance": model_sample["corrected_edit_distance"],
                        f"{alias}_corrected_result": model_sample["corrected_result"],
                        f"{alias}_corrected_result_match": model_sample["corrected_result_match"],
                    }
                )
            writer.writerow(row)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara dois ou mais modelos do MathSolverAI sobre o mesmo benchmark.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Manifesto .json, .jsonl ou .csv com image e expected_expression.",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Modelo no formato alias=caminho.pth. Pode repetir a flag.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Caminho opcional do relatorio JSON consolidado.",
    )
    parser.add_argument(
        "--csv-path",
        default="",
        help="Caminho opcional do CSV consolidado por amostra.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Quantidade de candidatos por simbolo consultada no benchmark.",
    )
    parser.add_argument(
        "--disable-correction",
        action="store_true",
        help="Desliga o corretor por top-k durante o benchmark.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=24,
        help="Largura do beam search do corretor.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="Quantidade maxima de candidatos mantidos por amostra.",
    )
    parser.add_argument(
        "--show-model-output",
        action="store_true",
        help="Mostra a saida detalhada do pipeline de cada modelo durante a comparacao.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    samples = load_benchmark_samples(args.manifest)
    model_specs = [parse_model_spec(spec) for spec in args.model]
    model_paths = dict(model_specs)
    model_reports: dict[str, dict[str, Any]] = {}

    for alias, model_path in model_specs:
        app = MathSolverAI(model_path)
        if args.show_model_output:
            report = evaluate_samples(
                app,
                samples,
                top_k=args.top_k,
                use_correction=not args.disable_correction,
                beam_width=args.beam_width,
                max_candidates=args.max_candidates,
            )
        else:
            with redirect_stdout(io.StringIO()):
                report = evaluate_samples(
                    app,
                    samples,
                    top_k=args.top_k,
                    use_correction=not args.disable_correction,
                    beam_width=args.beam_width,
                    max_candidates=args.max_candidates,
                )
        model_reports[alias] = report

    comparison = build_comparison_report(args.manifest, model_reports=model_reports, model_paths=model_paths)
    print_comparison_summary(comparison)

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"[INFO] Relatorio salvo em: {report_path}")
    if args.csv_path:
        write_comparison_csv(comparison, args.csv_path)
        print(f"[INFO] CSV salvo em: {args.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
