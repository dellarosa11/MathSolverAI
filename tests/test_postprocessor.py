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
