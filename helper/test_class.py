import pytest
import cv2
import numpy as np

from helper.helper_function import preprocess, check_device


@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


def test_preprocess(sample_image):
    w, h = 224, 224
    result = preprocess(sample_image, w, h)
    assert result.shape == (1, 3, w, h)


def test_check_device():
    result = check_device()
    assert result in ["CPU", "GPU"]
