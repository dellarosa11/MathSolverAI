from types import SimpleNamespace

from src.postprocessor import ExpressionCorrector
from src.solver import MathSolver


def _symbol(label, confidence, top_predictions):
    return SimpleNamespace(
        label=label,
        confidence=confidence,
        top_predictions=top_predictions,
    )


def test_corrector_prefers_valid_expression_over_raw_invalid_one():
    recognition = SimpleNamespace(
        symbols=[
            _symbol("1", 0.95, [{"label": "1", "confidence": 0.95}]),
            _symbol("=", 0.52, [{"label": "=", "confidence": 0.52}, {"label": "+", "confidence": 0.48}]),
            _symbol("2", 0.96, [{"label": "2", "confidence": 0.96}]),
            _symbol("=", 0.95, [{"label": "=", "confidence": 0.95}]),
            _symbol("3", 0.97, [{"label": "3", "confidence": 0.97}]),
        ]
    )

    corrector = ExpressionCorrector(MathSolver())
    correction = corrector.correct(recognition, beam_width=8, alternatives_per_symbol=2, max_candidates=5)

    assert correction.raw_expression == "1=2=3"
    assert correction.corrected_expression == "1+2=3"
    assert correction.changed is True
    assert correction.selected_candidate.valid is True


def test_corrector_identity_keeps_expression():
    corrector = ExpressionCorrector(MathSolver())

    correction = corrector.identity("1+2")

    assert correction.corrected_expression == "1+2"
    assert correction.changed is False


def test_corrector_prefers_true_equation_when_equals_has_lower_confidence():
    recognition = SimpleNamespace(
        symbols=[
            _symbol("1", 0.95, [{"label": "1", "confidence": 0.95}]),
            _symbol("3", 0.94, [{"label": "3", "confidence": 0.94}]),
            _symbol("+", 0.91, [{"label": "+", "confidence": 0.91}]),
            _symbol("5", 0.95, [{"label": "5", "confidence": 0.95}]),
            _symbol("7", 0.98, [{"label": "7", "confidence": 0.98}]),
            _symbol(
                "5",
                0.31,
                [
                    {"label": "5", "confidence": 0.31},
                    {"label": "2", "confidence": 0.29},
                    {"label": "=", "confidence": 0.16},
                ],
            ),
            _symbol("7", 0.97, [{"label": "7", "confidence": 0.97}]),
            _symbol("0", 0.96, [{"label": "0", "confidence": 0.96}]),
        ]
    )

    corrector = ExpressionCorrector(MathSolver())
    correction = corrector.correct(recognition, beam_width=12, alternatives_per_symbol=3, max_candidates=6)

    assert correction.raw_expression == "13+57570"
    assert correction.corrected_expression == "13+57=70"
    assert correction.selected_candidate.valid is True
    assert correction.selected_candidate.solvable is True


def test_corrector_prefers_multi_operator_expression_when_middle_symbol_is_ambiguous():
    recognition = SimpleNamespace(
        symbols=[
            _symbol("5", 0.96, [{"label": "5", "confidence": 0.96}]),
            _symbol("+", 0.93, [{"label": "+", "confidence": 0.93}]),
            _symbol("3", 0.95, [{"label": "3", "confidence": 0.95}]),
            _symbol(
                "4",
                0.40,
                [
                    {"label": "4", "confidence": 0.40},
                    {"label": "1", "confidence": 0.12},
                    {"label": "6", "confidence": 0.10},
                    {"label": "+", "confidence": 0.09},
                ],
            ),
            _symbol("2", 0.92, [{"label": "2", "confidence": 0.92}]),
        ]
    )
    recognition.symbols[3].box = (90, 23, 15, 16)

    corrector = ExpressionCorrector(MathSolver())
    correction = corrector.correct(recognition, beam_width=16, alternatives_per_symbol=4, max_candidates=8)

    assert correction.corrected_expression == "5+3+2"


def test_corrector_prefers_division_over_false_equation_for_ambiguous_bar():
    recognition = SimpleNamespace(
        symbols=[
            _symbol("4", 0.95, [{"label": "4", "confidence": 0.95}]),
            _symbol(
                "=",
                0.39,
                [
                    {"label": "=", "confidence": 0.39},
                    {"label": "/", "confidence": 0.36},
                    {"label": "-", "confidence": 0.08},
                ],
            ),
            _symbol("2", 0.94, [{"label": "2", "confidence": 0.94}]),
        ]
    )
    recognition.symbols[1].box = (30, 10, 30, 12)

    corrector = ExpressionCorrector(MathSolver())
    correction = corrector.correct(recognition, beam_width=8, alternatives_per_symbol=3, max_candidates=6)

    assert correction.corrected_expression == "4/2"


def test_corrector_builds_fraction_candidate_from_layout():
    recognition = SimpleNamespace(
        symbols=[
            _symbol("4", 0.96, [{"label": "4", "confidence": 0.96}]),
            _symbol("+", 0.93, [{"label": "+", "confidence": 0.93}]),
            _symbol("3", 0.95, [{"label": "3", "confidence": 0.95}]),
            _symbol(
                "-",
                0.95,
                [
                    {"label": "-", "confidence": 0.95},
                    {"label": "/", "confidence": 0.02},
                    {"label": "=", "confidence": 0.01},
                ],
            ),
            _symbol("4", 0.96, [{"label": "4", "confidence": 0.96}]),
        ]
    )
    recognition.symbols[0].box = (44, 0, 25, 29)
    recognition.symbols[1].box = (75, 11, 24, 18)
    recognition.symbols[2].box = (96, 4, 23, 33)
    recognition.symbols[3].box = (31, 30, 46, 8)
    recognition.symbols[4].box = (46, 38, 23, 30)

    corrector = ExpressionCorrector(MathSolver())
    correction = corrector.correct(recognition, beam_width=12, alternatives_per_symbol=3, max_candidates=8)

    assert correction.corrected_expression == "4/4+3"


def test_corrector_can_prefer_multiplication_for_compact_ambiguous_symbol():
    recognition = SimpleNamespace(
        symbols=[
            _symbol("5", 0.96, [{"label": "5", "confidence": 0.96}]),
            _symbol("5", 0.96, [{"label": "5", "confidence": 0.96}]),
            _symbol(
                "7",
                0.29,
                [
                    {"label": "7", "confidence": 0.29},
                    {"label": "2", "confidence": 0.22},
                    {"label": "1", "confidence": 0.15},
                    {"label": "+", "confidence": 0.03},
                ],
            ),
            _symbol("2", 0.95, [{"label": "2", "confidence": 0.95}]),
        ]
    )
    recognition.symbols[2].box = (72, 10, 18, 16)

    corrector = ExpressionCorrector(MathSolver())
    correction = corrector.correct(recognition, beam_width=16, alternatives_per_symbol=4, max_candidates=8)

    assert correction.corrected_expression == "55*2"


def test_corrector_can_expand_wide_zero_blob_into_double_zero():
    recognition = SimpleNamespace(
        symbols=[
            _symbol("1", 0.95, [{"label": "1", "confidence": 0.95}]),
            _symbol(
                "0",
                0.66,
                [
                    {"label": "0", "confidence": 0.66},
                    {"label": "6", "confidence": 0.09},
                    {"label": "8", "confidence": 0.05},
                ],
            ),
            _symbol("-", 0.97, [{"label": "-", "confidence": 0.97}]),
            _symbol("2", 0.95, [{"label": "2", "confidence": 0.95}]),
            _symbol("0", 0.93, [{"label": "0", "confidence": 0.93}]),
        ]
    )
    recognition.symbols[1].box = (21, 43, 56, 36)

    corrector = ExpressionCorrector(MathSolver())
    correction = corrector.correct(recognition, beam_width=16, alternatives_per_symbol=4, max_candidates=8)

    assert correction.corrected_expression == "100-20"
