from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from numpy import ndarray, uint8, zeros, float64, clip, array, zeros_like
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
from src.edge_processing import reflection_method, zero_method, extend_method, wrap_method

"python -m src.main image1.png -o output2.png -k emboss"
"python -m src.main image1.png -o output1.png -e zero"
"python -m src.main image1.png -o output3.png -k gaussian_blur -c"
"python -m src.main image1.png -o output4.png -k highlighting_horizontal_borders -c"


def get_image_path_for_read(image_name: str) -> Path:
    """
    Возвращает полный путь к изображению в папке images
    """
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    image_path = project_root / "images" / image_name
    return image_path


def get_image_path_for_save(image_name: str) -> Path:
    """
    Возвращает полный путь к изображению в папке results
    """
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    image_path = project_root / "results" / image_name
    return image_path


def get_kernel(kernel_name: str) -> ndarray:
    if kernel_name == "blur":
        kernel_list = blur_kernel
    elif kernel_name == "sharp":
        kernel_list = sharpness_kernel
    elif kernel_name == "emboss":
        kernel_list = emboss_kernel
    elif kernel_name == "gaussian_blur":
        kernel_list = gaussian_blur
    elif kernel_name == "gaussian_blur_5x5":
        kernel_list = gaussian_blur_5x5
    elif kernel_name == "highlighting_vertical_borders":
        kernel_list = highlighting_vertical_borders
    elif kernel_name == "highlighting_horizontal_borders":
        kernel_list = highlighting_horizontal_borders
    elif kernel_name == "box_blur_5x5":
        kernel_list = box_blur_5x5
    else:
        kernel_list = blur_kernel

    return array(kernel_list, dtype=float64)


def apply_convolution(image: ndarray, kernel: ndarray, edge_mode: str = "reflect") -> ndarray:
    if edge_mode == "reflect":
        edge_processing = reflection_method
    elif edge_mode == "zero":
        edge_processing = zero_method
    elif edge_mode == "extend":
        edge_processing = extend_method
    elif edge_mode == "wrap":
        edge_processing = wrap_method
    else:
        raise ValueError(f"Unknown edge mode {edge_mode}")

    kernel_h = kernel.shape[0]
    kernel_half = kernel_h // 2
    height, width = image.shape
    result = zeros((height, width), dtype=float64)

    for y in range(height):
        for x in range(width):
            pixel_sum = 0.0
            for ky in range(kernel_h):
                for kx in range(kernel_h):
                    px = x + kx - kernel_half
                    py = y + ky - kernel_half
                    coords = edge_processing(px, py, width, height)
                    if coords is not None:
                        pixel_x, pixel_y = coords
                        pixel_sum += image[pixel_y, pixel_x] * kernel[ky, kx]
            result[y, x] = pixel_sum
    return result


def apply_emboss(result: ndarray, kernel_name: str) -> ndarray:
    """
    Если выбрано ядро emboss, прибавляет 128 к каждому пикселю.
    """
    if kernel_name == "emboss":
        result = result + 128
    return result


def apply_convolution_rgb(image: ndarray, kernel: ndarray, edge_mode: str = "reflect") -> ndarray:
    """
    Применяет свёртку ко всем трём каналам RGB-изображения
    """
    height, width, c = image.shape
    result = zeros_like(image)
    for channel in range(c):
        result[:, :, channel] = apply_convolution(image[:, :, channel], kernel, edge_mode)
    return result


def rgb_to_grayscale(image: ndarray) -> ndarray:
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def main():
    parser = ArgumentParser(
        description="Image convolution with kernel selection and edge processing"
    )
    parser.add_argument("input", help="Input image from folder images")
    parser.add_argument(
        "--output", "-o", default="output.jpg", help="Output file (default output.jpg)"
    )
    parser.add_argument(
        "--kernel",
        "-k",
        choices=[
            "blur",
            "sharp",
            "emboss",
            "gaussian_blur",
            "highlighting_horizontal_borders",
            "box_blur_5x5",
            "highlighting_vertical_borders",
            "gaussian_blur_5x5",
        ],
        default="blur",
        help="Type kernel (blur, sharp, emboss)",
    )
    parser.add_argument(
        "--edge",
        "-e",
        choices=["zero", "reflect", "extend", "wrap"],
        default="reflect",
        help="Method of processing the edges",
    )
    parser.add_argument(
        "--color", "-c", action="store_true", help="Process RGB image (keep colors)"
    )
    args = parser.parse_args()
    image = array(Image.open(get_image_path_for_read(args.input)), dtype=float64)
    kernel = get_kernel(args.kernel)
    if args.color and image.ndim == 3 and image.shape[2] == 3:
        # Свертка RGB изображения
        result = apply_convolution_rgb(image, kernel, edge_mode=args.edge)
        result = apply_emboss(result, args.kernel)
    else:
        if image.ndim == 3:
            grayscale = rgb_to_grayscale(image)
        else:
            grayscale = image
        result = apply_convolution(grayscale, kernel, edge_mode=args.edge)

    result = clip(result, 0, 255)
    result = result.astype(uint8)
    gray_image = Image.fromarray(result)
    gray_image.save(get_image_path_for_save(args.output))


if __name__ == "__main__":
    main()
