from src.data.class_config import FOLDER_TO_INDEX, get_folder_name_for_label, get_label_for_folder_name


def test_digit_folders_are_supported_in_custom_dataset():
    assert "0" in FOLDER_TO_INDEX
    assert "9" in FOLDER_TO_INDEX


def test_label_and_folder_name_conversion_round_trip():
    assert get_folder_name_for_label("+") == "plus"
    assert get_folder_name_for_label("3") == "3"
    assert get_label_for_folder_name("minus") == "-"
    assert get_label_for_folder_name("7") == "7"
