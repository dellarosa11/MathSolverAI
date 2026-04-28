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


def test_prepare_for_nn_returns_28x28_image():
    roi = np.full((10, 20), 255, dtype=np.uint8)
    processor = ImageProcessor()

    prepared = processor.prepare_for_nn(roi)

    assert prepared.shape == (28, 28)
    assert prepared.dtype == np.uint8
    assert prepared.max() == 255
