from types import SimpleNamespace

import numpy as np
import pytest

from main import MathSolverAI


class DummyProcessor:
    def __init__(self, boxes, binary):
        self.boxes = boxes
        self.binary = binary

    def get_processed_pipeline(self, image_path):
        return self.binary.copy(), self.binary.copy()

    def extract_bounding_boxes(self, binary_img):
        return list(self.boxes)

    def prepare_for_nn(self, roi, target_size=28):
        return roi


class DummyPredictor:
    def __init__(self, labels_by_mean):
        self.labels_by_mean = labels_by_mean

    def predict_with_confidence(self, char_img, top_k=3):
        key = int(round(float(char_img.mean())))
        label = self.labels_by_mean[key]
        return type(
            "PredictionResult",
            (),
            {
                "label": label,
                "confidence": 0.95,
                "top_predictions": [{"label": label, "confidence": 0.95}],
            },
        )()

    def predict(self, char_img):
        return self.predict_with_confidence(char_img).label


class DummySolver:
    def __init__(self, result):
        self.result = result
        self.last_expression = None

    def normalize_expression(self, expression):
        return expression

    def solve(self, expression):
        self.last_expression = expression
        return self.result


class DummyCorrector:
    def correct(self, recognition, **kwargs):
        expression = recognition.expression
        candidate = SimpleNamespace(
            expression=expression,
            normalized_expression=expression,
            score=1.0,
            valid=True,
            solvable=True,
        )
        return SimpleNamespace(
            raw_expression=expression,
            corrected_expression=expression,
            changed=False,
            selected_candidate=candidate,
            candidates=[candidate],
        )

    def identity(self, expression):
        candidate = SimpleNamespace(
            expression=expression,
            normalized_expression=expression,
            score=1.0,
            valid=True,
            solvable=True,
        )
        return SimpleNamespace(
            raw_expression=expression,
            corrected_expression=expression,
            changed=False,
            selected_candidate=candidate,
            candidates=[candidate],
        )


def _build_test_app(binary, boxes, labels_by_mean, result):
    ai = MathSolverAI.__new__(MathSolverAI)
    ai.processor = DummyProcessor(boxes=boxes, binary=binary)
    ai.predictor = DummyPredictor(labels_by_mean)
    ai.solver = DummySolver(result=result)
    ai.corrector = DummyCorrector()
    return ai


def test_run_pipeline_builds_expression_in_left_to_right_order():
    binary = np.zeros((32, 32), dtype=np.uint8)
    binary[0:5, 0:5] = 10
    binary[0:5, 12:17] = 20
    binary[0:5, 20:25] = 30

    ai = _build_test_app(
        binary=binary,
        boxes=[(12, 0, 5, 5), (0, 0, 5, 5), (20, 0, 5, 5)],
        labels_by_mean={10: "1", 20: "+", 30: "2"},
        result=3,
    )

    result = ai.run_pipeline("ignored.png")

    assert result == "3"
    assert ai.solver.last_expression == "1+2"


def test_run_pipeline_can_return_only_the_expression():
    binary = np.zeros((32, 32), dtype=np.uint8)
    binary[0:5, 0:5] = 10
    binary[0:5, 8:13] = 20

    ai = _build_test_app(
        binary=binary,
        boxes=[(8, 0, 5, 5), (0, 0, 5, 5)],
        labels_by_mean={10: "3", 20: "4"},
        result=None,
    )

    result = ai.run_pipeline("ignored.png", solve_expression=False)

    assert result == "34"
    assert ai.solver.last_expression is None


def test_run_pipeline_returns_empty_string_when_no_boxes():
    ai = _build_test_app(
        binary=np.zeros((32, 32), dtype=np.uint8),
        boxes=[],
        labels_by_mean={},
        result=None,
    )

    result = ai.run_pipeline("ignored.png")

    assert result == ""
    assert ai.solver.last_expression is None


def test_validate_paths_accepts_existing_files(tmp_path):
    image_path = tmp_path / "input.png"
    model_path = tmp_path / "model.pth"
    image_path.write_bytes(b"fake-image")
    model_path.write_bytes(b"fake-model")

    resolved_image, resolved_model = MathSolverAI.validate_paths(image_path, model_path)

    assert resolved_image == image_path
    assert resolved_model == model_path


def test_validate_paths_rejects_missing_files(tmp_path):
    missing_image = tmp_path / "missing.png"
    missing_model = tmp_path / "missing.pth"

    with pytest.raises(FileNotFoundError):
        MathSolverAI.validate_paths(missing_image, missing_model)
