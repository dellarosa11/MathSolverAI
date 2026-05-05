from src.models.cnn_model import MathCNN, MathCNNPlus
from src.models.model_factory import build_model


def test_build_model_supports_legacy_cnn():
    model = build_model("cnn", num_classes=17)

    assert isinstance(model, MathCNN)


def test_build_model_supports_enhanced_cnn_aliases():
    model = build_model("cnn_plus", num_classes=17)
    alias_model = build_model("enhanced_cnn", num_classes=17)

    assert isinstance(model, MathCNNPlus)
    assert isinstance(alias_model, MathCNNPlus)
