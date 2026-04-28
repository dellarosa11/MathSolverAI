from src.data.generate_synthetic_symbols import discover_font_paths, expand_requested_labels


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
