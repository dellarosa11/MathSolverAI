from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from sympy import simplify


EPSILON = 1e-9
INVALID_START_TOKENS = {"*", "/", "=", ")"}
STRICT_OPERATORS = {"*", "/", "="}


@dataclass(frozen=True)
class ExpressionCandidate:
    expression: str
    normalized_expression: str | None
    score: float
    valid: bool
    solvable: bool


@dataclass(frozen=True)
class ExpressionCorrectionResult:
    raw_expression: str
    corrected_expression: str
    changed: bool
    selected_candidate: ExpressionCandidate
    candidates: list[ExpressionCandidate]


class ExpressionCorrector:
    """
    Ajusta a expressao reconhecida usando alternativas do top-k e regras leves.
    """

    def __init__(self, solver: Any):
        self.solver = solver

    def _equation_structure_bonus(self, normalized_expression: str) -> float:
        if "=" not in normalized_expression:
            return 0.0

        left_side, right_side = normalized_expression.split("=", 1)
        bonus = 0.6

        try:
            left_expr = self.solver._parse(left_side)
            right_expr = self.solver._parse(right_side)
            difference = simplify(left_expr - right_expr)
            free_symbols = getattr(difference, "free_symbols", set())

            if free_symbols:
                bonus += 0.18
            elif difference == 0:
                bonus += 0.55
            else:
                bonus -= 0.12
        except Exception:
            bonus -= 0.05

        return bonus

    @staticmethod
    def _plain_number_penalty(normalized_expression: str) -> float:
        if normalized_expression.isdigit() and len(normalized_expression) >= 5:
            return min(0.25, 0.05 * (len(normalized_expression) - 4))
        return 0.0

    @staticmethod
    def _extract_symbol_candidates(symbol: Any, limit: int) -> list[dict[str, float]]:
        top_predictions = getattr(symbol, "top_predictions", None) or []
        if not top_predictions:
            label = getattr(symbol, "label", "")
            confidence = float(getattr(symbol, "confidence", 1.0))
            return [{"label": label, "confidence": confidence}]

        unique_candidates: list[dict[str, float]] = []
        seen_labels: set[str] = set()
        for prediction in top_predictions:
            label = str(prediction["label"])
            if label in seen_labels:
                continue
            seen_labels.add(label)
            unique_candidates.append(
                {
                    "label": label,
                    "confidence": float(prediction["confidence"]),
                }
            )
            if len(unique_candidates) >= limit:
                break
        return unique_candidates

    @staticmethod
    def _partial_expression_penalty(expression: str) -> float | None:
        if not expression:
            return 0.0

        if expression[0] in INVALID_START_TOKENS:
            return None

        if expression.count("=") > 1:
            return None

        depth = 0
        previous = ""
        for index, token in enumerate(expression):
            if token == "(":
                depth += 1
            elif token == ")":
                depth -= 1
                if depth < 0:
                    return None

            if previous:
                if previous in STRICT_OPERATORS and token in STRICT_OPERATORS:
                    return None
                if previous == "(" and token in {"*", "/", "=", ")"}:
                    return None
                if previous == "=" and token in {"=", "*", "/", ")"}:
                    return None
                if previous in {"+", "-"} and token == "=":
                    return None
                if previous == "=" and token in {"+", "-"} and index == len(expression) - 1:
                    return None
            previous = token

        penalty = 0.0
        penalty -= depth * 0.03
        if expression.endswith(tuple(STRICT_OPERATORS)):
            penalty -= 0.2
        return penalty

    def _evaluate_expression(self, expression: str, base_score: float, is_raw: bool) -> ExpressionCandidate:
        score = base_score + (0.08 if is_raw else 0.0)
        normalized_expression: str | None = None
        valid = False
        solvable = False

        try:
            normalized_expression = self.solver.normalize_expression(expression)
            valid = True
            score += 0.35
            score += self._equation_structure_bonus(normalized_expression)
            score -= self._plain_number_penalty(normalized_expression)

            try:
                self.solver.solve(normalized_expression)
                solvable = True
                score += 0.2
            except Exception:
                score -= 0.03
        except Exception:
            score -= 0.2

        return ExpressionCandidate(
            expression=expression,
            normalized_expression=normalized_expression,
            score=score,
            valid=valid,
            solvable=solvable,
        )

    def _build_beam(
        self,
        symbols: Sequence[Any],
        *,
        alternatives_per_symbol: int,
        beam_width: int,
    ) -> list[tuple[str, float]]:
        beam: list[tuple[str, float]] = [("", 0.0)]

        for symbol in symbols:
            next_beam: list[tuple[str, float]] = []
            for expression, score in beam:
                for candidate in self._extract_symbol_candidates(symbol, alternatives_per_symbol):
                    next_expression = expression + candidate["label"]
                    penalty = self._partial_expression_penalty(next_expression)
                    if penalty is None:
                        continue

                    next_score = score + math.log(max(candidate["confidence"], EPSILON)) + penalty
                    next_beam.append((next_expression, next_score))

            if not next_beam:
                break

            next_beam.sort(key=lambda item: item[1], reverse=True)
            beam = next_beam[:beam_width]

        return beam

    @staticmethod
    def _dedupe_candidates(candidates: Iterable[ExpressionCandidate]) -> list[ExpressionCandidate]:
        best_by_expression: dict[str, ExpressionCandidate] = {}
        for candidate in candidates:
            existing = best_by_expression.get(candidate.expression)
            if existing is None or candidate.score > existing.score:
                best_by_expression[candidate.expression] = candidate
        return sorted(best_by_expression.values(), key=lambda item: item.score, reverse=True)

    def correct(
        self,
        recognition: Any,
        *,
        beam_width: int = 24,
        alternatives_per_symbol: int = 3,
        max_candidates: int = 8,
    ) -> ExpressionCorrectionResult:
        raw_expression = "".join(getattr(symbol, "label", "") for symbol in getattr(recognition, "symbols", []))
        raw_score = 0.0
        for symbol in getattr(recognition, "symbols", []):
            raw_score += math.log(max(float(getattr(symbol, "confidence", 1.0)), EPSILON))

        beam = self._build_beam(
            getattr(recognition, "symbols", []),
            alternatives_per_symbol=alternatives_per_symbol,
            beam_width=beam_width,
        )

        evaluated_candidates = [
            self._evaluate_expression(expression, base_score, is_raw=(expression == raw_expression))
            for expression, base_score in beam
        ]
        evaluated_candidates.append(self._evaluate_expression(raw_expression, raw_score, is_raw=True))

        ranked_candidates = self._dedupe_candidates(evaluated_candidates)[:max_candidates]
        selected = ranked_candidates[0]

        return ExpressionCorrectionResult(
            raw_expression=raw_expression,
            corrected_expression=selected.expression,
            changed=selected.expression != raw_expression,
            selected_candidate=selected,
            candidates=ranked_candidates,
        )

    def identity(self, expression: str) -> ExpressionCorrectionResult:
        candidate = self._evaluate_expression(expression, base_score=0.0, is_raw=True)
        return ExpressionCorrectionResult(
            raw_expression=expression,
            corrected_expression=expression,
            changed=False,
            selected_candidate=candidate,
            candidates=[candidate],
        )
