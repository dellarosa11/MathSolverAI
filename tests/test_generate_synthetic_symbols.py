import random

import numpy as np

from src.data.generate_synthetic_symbols import (
    discover_font_paths,
    expand_requested_labels,
    render_symbol_image,
)


def test_expand_requested_labels_supports_keywords_and_folder_names():
    labels = expand_requested_labels(["digits", "plus", "operators"])

    assert "0" in labels
    assert "9" in labels
    assert "+" in labels
    assert "-" in labels


def test_discover_font_paths_filters_supported_extensions(tmp_path):
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    (fonts_dir / "fake_font.ttf").write_text("font", encoding="utf-8")
    (fonts_dir / "other_font.otf").write_text("font", encoding="utf-8")
    (fonts_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    paths = discover_font_paths(font_dirs=[fonts_dir], limit=10)

    names = {path.name for path in paths}
    assert "fake_font.ttf" in names
    assert "other_font.otf" in names
    assert "notes.txt" not in names


def test_render_symbol_image_generates_pixels_for_manual_operator_variants():
    rng = random.Random(7)

    times_image = render_symbol_image("*", font_paths=[], rng=rng, canvas_size=96)
    div_image = render_symbol_image("/", font_paths=[], rng=rng, canvas_size=96)

    assert np.array(times_image).max() > 0
    assert np.array(div_image).max() > 0
