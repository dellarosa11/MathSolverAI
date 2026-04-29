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
