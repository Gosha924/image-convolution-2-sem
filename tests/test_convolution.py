import os
import pytest
import numpy as np
from PIL import Image
from src.main import apply_convolution, apply_convolution_rgb, rgb_to_grayscale
from src.kernels import (
    blur_kernel,
    emboss_kernel,
    sharpness_kernel,
    gaussian_blur,
    gaussian_blur_5x5,
    highlighting_vertical_borders,
    highlighting_horizontal_borders,
    box_blur_5x5,
)

GOLDEN_DIR = "tests/golden_images"
TEST_IMAGE_PATH = "images/image1.png"
"pytest tests/test_convolution.py -v"

KERNELS = {
    "blur": blur_kernel,
    "sharp": sharpness_kernel,
    "emboss": emboss_kernel,
    "gaussian_blur": gaussian_blur,
    "gaussian_blur_5x5": gaussian_blur_5x5,
    "highlighting_vertical_borders": highlighting_vertical_borders,
    "highlighting_horizontal_borders": highlighting_horizontal_borders,
    "box_blur": box_blur_5x5,
}

BORDER_MODES = ["zero", "reflect", "extend"]


@pytest.fixture(scope="module")
def input_image():
    img = Image.open(TEST_IMAGE_PATH).convert("RGB")
    return np.array(img, dtype=np.float64)


def test_grayscale_convolution(input_image):
    gray = rgb_to_grayscale(input_image)
    for kernel_name, kernel_list in KERNELS.items():
        kernel = np.array(kernel_list, dtype=np.float64)
        for mode in BORDER_MODES:
            result = apply_convolution(gray, kernel, edge_mode=mode)
            if kernel_name == "emboss":
                result += 128
            result = np.clip(result, 0, 255).astype(np.uint8)
            golden_path = os.path.join(GOLDEN_DIR, f"{kernel_name}_{mode}_gray.png")
            golden = np.array(Image.open(golden_path))
            np.testing.assert_allclose(
                result, golden, atol=1, err_msg=f"{kernel_name}_{mode}_gray"
            )


def test_rgb_convolution(input_image):
    for kernel_name, kernel_list in KERNELS.items():
        kernel = np.array(kernel_list, dtype=np.float64)
        for mode in BORDER_MODES:
            result = apply_convolution_rgb(input_image, kernel, edge_mode=mode)
            if kernel_name == "emboss":
                result += 128
            result = np.clip(result, 0, 255).astype(np.uint8)
            golden_path = os.path.join(GOLDEN_DIR, f"{kernel_name}_{mode}_color.png")
            golden = np.array(Image.open(golden_path))
            np.testing.assert_allclose(
                result, golden, atol=1, err_msg=f"{kernel_name}_{mode}_color"
            )
