from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from sympy import simplify


EPSILON = 1e-9
INVALID_START_TOKENS = {"*", "/", "=", ")"}
STRICT_OPERATORS = {"*", "/", "="}
BINARY_OPERATORS = {"+", "-", "*", "/"}
ALL_OPERATORS = BINARY_OPERATORS | {"="}


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
        bonus = 0.1

        try:
            left_expr = self.solver._parse(left_side)
            right_expr = self.solver._parse(right_side)
            difference = simplify(left_expr - right_expr)
            free_symbols = getattr(difference, "free_symbols", set())

            if free_symbols:
                bonus += 0.28
            elif difference == 0:
                bonus += 0.55
            else:
                bonus -= 0.25
        except Exception:
            bonus -= 0.05

        return bonus

    @staticmethod
    def _plain_number_penalty(normalized_expression: str) -> float:
        if normalized_expression.isdigit() and len(normalized_expression) >= 4:
            return min(0.35, 0.12 + (0.06 * (len(normalized_expression) - 4)))
        return 0.0

    @staticmethod
    def _upsert_candidate(
        candidates: list[dict[str, float]],
        *,
        label: str,
        confidence: float,
    ) -> None:
        bounded_confidence = float(max(EPSILON, min(0.99, confidence)))
        for candidate in candidates:
            if candidate["label"] == label:
                candidate["confidence"] = max(float(candidate["confidence"]), bounded_confidence)
                return
        candidates.append({"label": label, "confidence": bounded_confidence})

    def _apply_shape_priors(self, symbol: Any, candidates: list[dict[str, float]]) -> list[dict[str, float]]:
        box = getattr(symbol, "box", None)
        if not box:
            return candidates

        _, _, width, height = box
        if width <= 0 or height <= 0:
            return candidates

        aspect_ratio = width / max(1, height)
        best_confidence = max(float(candidate["confidence"]) for candidate in candidates)
        compact_symbol = 0.65 <= aspect_ratio <= 1.35 and max(width, height) <= 24
        flat_symbol = aspect_ratio >= 1.8 and height <= 18
        low_confidence = best_confidence < 0.7

        adjusted_candidates = [dict(candidate) for candidate in candidates]
        existing_by_label = {candidate["label"]: float(candidate["confidence"]) for candidate in adjusted_candidates}

        if compact_symbol and low_confidence:
            plus_signal = existing_by_label.get("+", 0.0)
            times_signal = existing_by_label.get("*", 0.0)
            if plus_signal >= 0.08:
                plus_confidence = max(0.3, plus_signal + 0.22)
                times_confidence = max(0.24, times_signal + 0.18)
            else:
                plus_confidence = max(0.24, plus_signal + 0.18)
                times_confidence = max(0.33, times_signal + 0.24)
            self._upsert_candidate(adjusted_candidates, label="+", confidence=plus_confidence)
            self._upsert_candidate(adjusted_candidates, label="*", confidence=times_confidence)

        if flat_symbol:
            base_confidence = 0.28 if low_confidence else 0.16
            self._upsert_candidate(
                adjusted_candidates,
                label="-",
                confidence=max(base_confidence, existing_by_label.get("-", 0.0) + 0.18),
            )
            self._upsert_candidate(
                adjusted_candidates,
                label="=",
                confidence=max(base_confidence - 0.03, existing_by_label.get("=", 0.0) + 0.16),
            )
            self._upsert_candidate(
                adjusted_candidates,
                label="/",
                confidence=max(base_confidence - 0.06, existing_by_label.get("/", 0.0) + 0.16),
            )

        adjusted_candidates.sort(key=lambda item: float(item["confidence"]), reverse=True)
        return adjusted_candidates

    def _apply_multi_character_priors(self, symbol: Any, candidates: list[dict[str, float]]) -> list[dict[str, float]]:
        box = getattr(symbol, "box", None)
        if not box:
            return candidates

        _, _, width, height = box
        if width <= 0 or height <= 0:
            return candidates

        adjusted_candidates = [dict(candidate) for candidate in candidates]
        existing_by_label = {candidate["label"]: float(candidate["confidence"]) for candidate in adjusted_candidates}
        zero_confidence = existing_by_label.get("0", 0.0)

        if width >= 40 and (width / max(1, height)) >= 1.4 and zero_confidence >= 0.25:
            self._upsert_candidate(
                adjusted_candidates,
                label="00",
                confidence=max(0.78, zero_confidence * 1.08),
            )

        adjusted_candidates.sort(key=lambda item: float(item["confidence"]), reverse=True)
        return adjusted_candidates

    def _extract_symbol_candidates(self, symbol: Any, limit: int) -> list[dict[str, float]]:
        top_predictions = getattr(symbol, "top_predictions", None) or []
        if not top_predictions:
            label = getattr(symbol, "label", "")
            confidence = float(getattr(symbol, "confidence", 1.0))
            candidates = self._apply_shape_priors(symbol, [{"label": label, "confidence": confidence}])
            return self._apply_multi_character_priors(symbol, candidates)[:limit]

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
        candidates = self._apply_shape_priors(symbol, unique_candidates)
        return self._apply_multi_character_priors(symbol, candidates)[:limit]

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

    @staticmethod
    def _tokenize_expression(normalized_expression: str) -> list[str]:
        tokens: list[str] = []
        current_digits = ""

        for character in normalized_expression:
            if character.isdigit():
                current_digits += character
                continue

            if current_digits:
                tokens.append(current_digits)
                current_digits = ""
            tokens.append(character)

        if current_digits:
            tokens.append(current_digits)
        return tokens

    @staticmethod
    def _is_operand_token(token: str) -> bool:
        return token.isdigit() or token == "x" or token == ")" or token == "("

    def _expression_pattern_score(self, normalized_expression: str) -> float:
        tokens = self._tokenize_expression(normalized_expression)
        if not tokens:
            return 0.0

        score = 0.0
        operator_count = sum(token in ALL_OPERATORS for token in tokens)
        numeric_tokens = [token for token in tokens if token.isdigit()]

        for index, token in enumerate(tokens):
            if token in BINARY_OPERATORS and 0 < index < len(tokens) - 1:
                left = tokens[index - 1]
                right = tokens[index + 1]
                if self._is_operand_token(left) and self._is_operand_token(right):
                    score += 0.18
            elif token == "=" and 0 < index < len(tokens) - 1:
                if tokens[index - 1] != "(" and tokens[index + 1] != ")":
                    score += 0.14

        clean_alternation = False
        if operator_count >= 2:
            clean_alternation = True
            expecting_operand = True
            for token in tokens:
                if token in {"(", ")"}:
                    continue
                if expecting_operand:
                    if not self._is_operand_token(token):
                        clean_alternation = False
                        break
                else:
                    if token not in ALL_OPERATORS:
                        clean_alternation = False
                        break
                expecting_operand = not expecting_operand

            if clean_alternation:
                score += 0.42 + (0.16 * (operator_count - 2))

        if operator_count == 1 and len(numeric_tokens) == 2:
            shorter_operand = min(len(token) for token in numeric_tokens)
            longer_operand = max(len(token) for token in numeric_tokens)
            if shorter_operand == 1 and longer_operand >= 3:
                score -= 0.48

        if operator_count >= 2 and any(len(token) >= 3 for token in numeric_tokens):
            score -= 0.2

        for token in numeric_tokens:
            if len(token) > 1 and token.startswith("0"):
                score -= 0.32

        return score

    @staticmethod
    def _format_fraction_expression(numerator: str, denominator: str) -> str:
        if len(numerator) > 1:
            numerator = f"({numerator})"
        if len(denominator) > 1:
            denominator = f"({denominator})"
        return f"{numerator}/{denominator}"

    @staticmethod
    def _looks_like_fraction_bar(box: tuple[int, int, int, int]) -> bool:
        _, _, width, height = box
        return width >= 18 and height <= 12 and (width / max(1, height)) >= 2.4

    @staticmethod
    def _sort_boxes_reading_order(items: list[tuple[tuple[int, int, int, int], str]]) -> list[tuple[tuple[int, int, int, int], str]]:
        if not items:
            return []

        average_height = sum(box[3] for box, _ in items) / len(items)
        line_threshold = max(6, average_height * 0.6)
        sorted_items = sorted(items, key=lambda item: (item[0][1], item[0][0]))
        lines: list[list[tuple[tuple[int, int, int, int], str]]] = []

        for item in sorted_items:
            box, _ = item
            center_y = box[1] + box[3] / 2
            for line in lines:
                reference_box = line[0][0]
                reference_center_y = reference_box[1] + reference_box[3] / 2
                if abs(center_y - reference_center_y) <= line_threshold:
                    line.append(item)
                    break
            else:
                lines.append([item])

        ordered: list[tuple[tuple[int, int, int, int], str]] = []
        for line in lines:
            ordered.extend(sorted(line, key=lambda item: item[0][0]))
        return ordered

    def _derive_fraction_candidate(self, recognition: Any, raw_score: float) -> tuple[str, float] | None:
        symbols = list(getattr(recognition, "symbols", []))
        if len(symbols) < 3:
            return None

        for index, symbol in enumerate(symbols):
            box = getattr(symbol, "box", None)
            if not box or not self._looks_like_fraction_bar(box):
                continue

            x, y, width, height = box
            center_y = y + height / 2
            numerator_indexes: list[int] = []
            denominator_indexes: list[int] = []

            for other_index, other_symbol in enumerate(symbols):
                if other_index == index:
                    continue
                other_box = getattr(other_symbol, "box", None)
                if not other_box:
                    continue

                ox, oy, ow, oh = other_box
                overlap = max(0, min(x + width, ox + ow) - max(x, ox))
                other_center_x = ox + ow / 2
                if not (x - 4 <= other_center_x <= x + width + 4) and overlap < max(6, ow * 0.35):
                    continue

                other_center_y = oy + oh / 2
                if other_center_y < center_y - max(6, height * 0.5):
                    numerator_indexes.append(other_index)
                elif other_center_y > center_y + max(6, height * 0.5):
                    denominator_indexes.append(other_index)

            if not numerator_indexes or not denominator_indexes:
                continue

            numerator_indexes.sort(key=lambda item: getattr(symbols[item], "box")[0])
            denominator_indexes.sort(key=lambda item: getattr(symbols[item], "box")[0])
            used_indexes = set(numerator_indexes + denominator_indexes + [index])

            numerator = "".join(getattr(symbols[item], "label", "") for item in numerator_indexes)
            denominator = "".join(getattr(symbols[item], "label", "") for item in denominator_indexes)
            if not numerator or not denominator:
                continue

            merged_box = (
                min(getattr(symbols[item], "box")[0] for item in used_indexes),
                min(getattr(symbols[item], "box")[1] for item in used_indexes),
                max(getattr(symbols[item], "box")[0] + getattr(symbols[item], "box")[2] for item in used_indexes)
                - min(getattr(symbols[item], "box")[0] for item in used_indexes),
                max(getattr(symbols[item], "box")[1] + getattr(symbols[item], "box")[3] for item in used_indexes)
                - min(getattr(symbols[item], "box")[1] for item in used_indexes),
            )
            items: list[tuple[tuple[int, int, int, int], str]] = [(merged_box, self._format_fraction_expression(numerator, denominator))]
            for other_index, other_symbol in enumerate(symbols):
                if other_index in used_indexes:
                    continue
                other_box = getattr(other_symbol, "box", None)
                if not other_box:
                    continue
                items.append((other_box, getattr(other_symbol, "label", "")))

            items = self._sort_boxes_reading_order(items)
            expression = "".join(text for _, text in items)
            return expression, raw_score + 0.65

        return None

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
            score += self._expression_pattern_score(normalized_expression)

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
        fraction_candidate = self._derive_fraction_candidate(recognition, raw_score)
        if fraction_candidate is not None:
            expression, score = fraction_candidate
            evaluated_candidates.append(self._evaluate_expression(expression, score, is_raw=False))

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
