from src.utils.export_inference_debug import slugify_label


def test_slugify_label_maps_math_symbols_to_safe_names():
    assert slugify_label("+") == "plus"
    assert slugify_label("(") == "lparen"
    assert slugify_label("7") == "7"
