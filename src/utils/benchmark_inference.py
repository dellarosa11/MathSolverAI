from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from main import MathSolverAI


@dataclass(frozen=True)
class BenchmarkSample:
    image_path: Path
    expected_expression: str
    expected_result: str | None = None
    sample_id: str | None = None


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _normalize_sample_entry(entry: dict[str, Any], base_dir: Path, index: int) -> BenchmarkSample:
    image_value = entry.get("image") or entry.get("image_path")
    expected_expression = entry.get("expected_expression") or entry.get("expression")
    expected_result = entry.get("expected_result")
    sample_id = entry.get("id") or entry.get("sample_id") or f"sample_{index:04d}"

    if not image_value or not expected_expression:
        raise ValueError("Cada amostra precisa ter 'image' e 'expected_expression'.")

    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = (base_dir / image_path).resolve()

    return BenchmarkSample(
        image_path=image_path,
        expected_expression=str(expected_expression).replace(" ", ""),
        expected_result=None if expected_result is None else str(expected_result),
        sample_id=str(sample_id),
    )


def load_benchmark_samples(manifest_path: str | Path) -> list[BenchmarkSample]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            "Manifesto de benchmark nao encontrado: "
            f"{path}. Crie um arquivo .json, .jsonl ou .csv com campos 'image' e 'expected_expression'."
        )

    base_dir = path.parent
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        entries = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            entries = payload.get("samples", [])
        elif isinstance(payload, list):
            entries = payload
        else:
            raise ValueError("Manifesto JSON precisa ser lista ou objeto com chave 'samples'.")
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            entries = list(csv.DictReader(handle))
    else:
        raise ValueError("Formato de manifesto nao suportado. Use .json, .jsonl ou .csv.")

    return [
        _normalize_sample_entry(entry, base_dir=base_dir, index=index)
        for index, entry in enumerate(entries, start=1)
    ]


def _safe_solve(app: MathSolverAI, expression: str) -> str | None:
    try:
        normalized = app.solver.normalize_expression(expression)
        return str(app.solver.solve(normalized))
    except Exception:
        return None


def evaluate_samples(
    app: MathSolverAI,
    samples: Iterable[BenchmarkSample],
    *,
    top_k: int = 3,
    use_correction: bool = True,
    beam_width: int = 24,
    max_candidates: int = 8,
) -> dict[str, Any]:
    sample_reports: list[dict[str, Any]] = []
    total_reference_symbols = 0
    raw_total_distance = 0
    corrected_total_distance = 0
    raw_exact_matches = 0
    corrected_exact_matches = 0
    raw_result_matches = 0
    corrected_result_matches = 0
    total_result_samples = 0
    improved_expression_samples = 0
    improved_result_samples = 0

    for sample in samples:
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Imagem do benchmark nao encontrada: {sample.image_path}")

        recognition = app.recognize_expression(sample.image_path, top_k=top_k)
        correction = app.improve_expression(
            recognition,
            use_correction=use_correction,
            beam_width=beam_width,
            alternatives_per_symbol=top_k,
            max_candidates=max_candidates,
        )

        raw_expression = recognition.expression
        corrected_expression = correction.corrected_expression
        raw_distance = levenshtein_distance(raw_expression, sample.expected_expression)
        corrected_distance = levenshtein_distance(corrected_expression, sample.expected_expression)

        total_reference_symbols += max(1, len(sample.expected_expression))
        raw_total_distance += raw_distance
        corrected_total_distance += corrected_distance
        raw_exact_matches += int(raw_expression == sample.expected_expression)
        corrected_exact_matches += int(corrected_expression == sample.expected_expression)

        raw_result = _safe_solve(app, raw_expression) if raw_expression else None
        corrected_result = _safe_solve(app, corrected_expression) if corrected_expression else None

        raw_result_match = None
        corrected_result_match = None
        if sample.expected_result is not None:
            total_result_samples += 1
            raw_result_match = raw_result == sample.expected_result
            corrected_result_match = corrected_result == sample.expected_result
            raw_result_matches += int(bool(raw_result_match))
            corrected_result_matches += int(bool(corrected_result_match))
            improved_result_samples += int(bool(corrected_result_match) and not bool(raw_result_match))

        improved_expression_samples += int(
            corrected_distance < raw_distance
            or (corrected_expression == sample.expected_expression and raw_expression != sample.expected_expression)
        )

        sample_reports.append(
            {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "expected_expression": sample.expected_expression,
                "raw_expression": raw_expression,
                "corrected_expression": corrected_expression,
                "raw_exact_match": raw_expression == sample.expected_expression,
                "corrected_exact_match": corrected_expression == sample.expected_expression,
                "raw_edit_distance": raw_distance,
                "corrected_edit_distance": corrected_distance,
                "expected_result": sample.expected_result,
                "raw_result": raw_result,
                "corrected_result": corrected_result,
                "raw_result_match": raw_result_match,
                "corrected_result_match": corrected_result_match,
                "correction_changed_expression": correction.changed,
                "correction_improved_expression": corrected_distance < raw_distance,
                "correction_improved_result": None
                if raw_result_match is None
                else bool(corrected_result_match) and not bool(raw_result_match),
                "correction_candidates": [
                    asdict(candidate) if is_dataclass(candidate) else dict(vars(candidate))
                    for candidate in correction.candidates
                ],
            }
        )

    sample_count = len(sample_reports)
    if sample_count == 0:
        raise ValueError("Nenhuma amostra encontrada no benchmark.")

    summary = {
        "sample_count": sample_count,
        "raw_expression_accuracy": raw_exact_matches / sample_count,
        "corrected_expression_accuracy": corrected_exact_matches / sample_count,
        "raw_symbol_accuracy": max(0.0, 1.0 - (raw_total_distance / total_reference_symbols)),
        "corrected_symbol_accuracy": max(0.0, 1.0 - (corrected_total_distance / total_reference_symbols)),
        "result_sample_count": total_result_samples,
        "raw_result_accuracy": None if total_result_samples == 0 else raw_result_matches / total_result_samples,
        "corrected_result_accuracy": None if total_result_samples == 0 else corrected_result_matches / total_result_samples,
        "samples_improved_by_correction": improved_expression_samples,
        "result_samples_improved_by_correction": improved_result_samples,
    }

    return {
        "summary": summary,
        "samples": sample_reports,
    }


def print_benchmark_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("[INFO] Resumo do benchmark")
    print(f"  Amostras: {summary['sample_count']}")
    print(f"  Raw expression accuracy: {summary['raw_expression_accuracy'] * 100:.2f}%")
    print(f"  Corrected expression accuracy: {summary['corrected_expression_accuracy'] * 100:.2f}%")
    print(f"  Raw symbol accuracy: {summary['raw_symbol_accuracy'] * 100:.2f}%")
    print(f"  Corrected symbol accuracy: {summary['corrected_symbol_accuracy'] * 100:.2f}%")
    print(f"  Samples improved by correction: {summary['samples_improved_by_correction']}")

    if summary["raw_result_accuracy"] is not None:
        print(f"  Raw result accuracy: {summary['raw_result_accuracy'] * 100:.2f}%")
        print(f"  Corrected result accuracy: {summary['corrected_result_accuracy'] * 100:.2f}%")
        print(
            "  Result samples improved by correction: "
            f"{summary['result_samples_improved_by_correction']}"
        )


def write_benchmark_csv(report: dict[str, Any], csv_path: str | Path) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = report["samples"]
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "sample_id",
        "image_path",
        "expected_expression",
        "raw_expression",
        "corrected_expression",
        "raw_exact_match",
        "corrected_exact_match",
        "raw_edit_distance",
        "corrected_edit_distance",
        "expected_result",
        "raw_result",
        "corrected_result",
        "raw_result_match",
        "corrected_result_match",
        "correction_changed_expression",
        "correction_improved_expression",
        "correction_improved_result",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa benchmark de inferencia em imagens rotuladas do MathSolverAI.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Manifesto .json, .jsonl ou .csv com image e expected_expression.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Checkpoint treinado usado no benchmark.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Caminho opcional do relatorio JSON de saida.",
    )
    parser.add_argument(
        "--csv-path",
        default="",
        help="Caminho opcional do relatorio CSV de saida por amostra.",
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
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    samples = load_benchmark_samples(args.manifest)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

    app = MathSolverAI(model_path)
    report = evaluate_samples(
        app,
        samples,
        top_k=args.top_k,
        use_correction=not args.disable_correction,
        beam_width=args.beam_width,
        max_candidates=args.max_candidates,
    )
    print_benchmark_summary(report)

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[INFO] Relatorio salvo em: {report_path}")
    if args.csv_path:
        write_benchmark_csv(report, args.csv_path)
        print(f"[INFO] CSV salvo em: {args.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
