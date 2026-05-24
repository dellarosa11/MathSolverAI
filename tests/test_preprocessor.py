import cv2
import numpy as np

from src.preprocessor import ImageProcessor


def test_extract_bounding_boxes_returns_left_to_right_order():
    binary = np.zeros((40, 80), dtype=np.uint8)
    cv2.rectangle(binary, (40, 8), (52, 30), 255, -1)
    cv2.rectangle(binary, (8, 8), (20, 30), 255, -1)

    processor = ImageProcessor()
    boxes = processor.extract_bounding_boxes(binary)

    assert len(boxes) == 2
    assert boxes[0][0] < boxes[1][0]


def test_extract_bounding_boxes_merges_fragmented_equals_sign():
    binary = np.zeros((40, 40), dtype=np.uint8)
    cv2.rectangle(binary, (8, 10), (24, 13), 255, -1)
    cv2.rectangle(binary, (8, 18), (24, 21), 255, -1)

    processor = ImageProcessor()
    boxes = processor.extract_bounding_boxes(binary)

    assert len(boxes) == 1
    assert boxes[0][2] >= 16


def test_extract_bounding_boxes_merges_fragmented_division_sign():
    binary = np.zeros((80, 80), dtype=np.uint8)
    cv2.rectangle(binary, (33, 10), (38, 15), 255, -1)
    cv2.rectangle(binary, (20, 30), (50, 35), 255, -1)
    cv2.rectangle(binary, (33, 38), (38, 43), 255, -1)

    processor = ImageProcessor()
    boxes = processor.extract_bounding_boxes(binary)

    assert len(boxes) == 1
    assert boxes[0][3] >= 30


def test_extract_bounding_boxes_does_not_merge_tall_stacked_symbols():
    binary = np.zeros((90, 40), dtype=np.uint8)
    cv2.rectangle(binary, (10, 8), (24, 30), 255, -1)
    cv2.rectangle(binary, (10, 48), (24, 70), 255, -1)

    processor = ImageProcessor()
    boxes = processor.extract_bounding_boxes(binary)

    assert len(boxes) == 2


def test_extract_bounding_boxes_filters_tiny_noise_blobs():
    binary = np.zeros((80, 80), dtype=np.uint8)
    cv2.rectangle(binary, (8, 8), (24, 36), 255, -1)
    cv2.rectangle(binary, (40, 10), (56, 34), 255, -1)
    cv2.rectangle(binary, (30, 60), (34, 64), 255, -1)
    cv2.rectangle(binary, (62, 62), (66, 66), 255, -1)

    processor = ImageProcessor()
    boxes = processor.extract_bounding_boxes(binary)

    assert len(boxes) == 2


def test_extract_bounding_boxes_splits_wide_connected_digit_blob():
    binary = np.zeros((50, 90), dtype=np.uint8)
    cv2.rectangle(binary, (8, 10), (28, 38), 255, -1)
    cv2.rectangle(binary, (40, 10), (60, 38), 255, -1)
    cv2.rectangle(binary, (29, 22), (39, 24), 255, -1)

    processor = ImageProcessor()
    boxes = processor.extract_bounding_boxes(binary)

    assert len(boxes) == 2
    assert boxes[0][0] < boxes[1][0]


def test_prepare_for_nn_returns_28x28_image():
    roi = np.full((10, 20), 255, dtype=np.uint8)
    processor = ImageProcessor()

    prepared = processor.prepare_for_nn(roi)

    assert prepared.shape == (28, 28)
    assert prepared.dtype == np.uint8
    assert prepared.max() == 255


def test_get_processing_debug_returns_notebook_treatment_stages(tmp_path):
    image = np.full((120, 220), 220, dtype=np.uint8)
    cv2.line(image, (10, 30), (210, 30), 180, 2)
    cv2.line(image, (10, 60), (210, 60), 180, 2)
    cv2.putText(image, "13+57=70", (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 35, 3, cv2.LINE_AA)

    image_path = tmp_path / "caderno.png"
    cv2.imwrite(str(image_path), image)

    processor = ImageProcessor(image_path)
    debug = processor.get_processing_debug()

    expected_keys = {
        "original",
        "denoised",
        "contrast",
        "line_mask",
        "line_removed",
        "normalized",
        "blurred",
        "threshold",
        "opened",
        "closed",
        "final_binary",
    }

    assert expected_keys.issubset(debug.keys())
    for stage in expected_keys:
        assert debug[stage].shape == image.shape
        assert debug[stage].dtype == np.uint8
    assert cv2.countNonZero(debug["line_mask"]) > 0
    assert int(debug["line_removed"][30, 50]) > int(debug["contrast"][30, 50])
    assert debug["final_binary"].max() == 255
