import random

import numpy as np
from PIL import Image

from src.data.dataset_builder import NotebookPhotoAugmentation, get_base_transform


def test_notebook_photo_augmentation_preserves_image_shape():
    random.seed(123)
    base = Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode="L")
    augmenter = NotebookPhotoAugmentation(line_probability=1.0)

    augmented = augmenter(base)

    assert augmented.size == (28, 28)
    assert np.asarray(augmented).dtype == np.uint8
    assert np.asarray(augmented).max() > 0


def test_base_transform_with_augmentation_returns_tensor():
    random.seed(123)
    np.random.seed(123)
    base = Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode="L")
    transform = get_base_transform(train=True, use_augmentation=True)

    tensor = transform(base)

    assert tensor.shape == (1, 28, 28)
    assert tensor.dtype.is_floating_point
