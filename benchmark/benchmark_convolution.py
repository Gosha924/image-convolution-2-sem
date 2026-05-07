import json
import time
import platform
import psutil
from pathlib import Path
import numpy as np
import cv2
import csv
from src.main import apply_convolution, apply_convolution_rgb
from src.kernels import (
    blur_kernel,
    emboss_kernel,
    sharpness_kernel,
    gaussian_blur,
    highlighting_vertical_borders,
    highlighting_horizontal_borders,
    box_blur_5x5,
    gaussian_blur_5x5,
)

SIZES = [128, 256, 512]
NUM_RUNS = 5
WARMUP = 2

KERNEL_FUNCS = {
    "blur_3x3": blur_kernel,
    "sharp_3x3": sharpness_kernel,
    "emboss_3x3": emboss_kernel,
    "gaussian_blur_3x3": gaussian_blur,
    "vert_borders": highlighting_vertical_borders,
    "horiz_borders": highlighting_horizontal_borders,
    "box_blur_5x5": box_blur_5x5,
    "gaussian_blur_5x5": gaussian_blur_5x5,
}

OPENCV_BORDER = {
    "zero": cv2.BORDER_CONSTANT,
    "reflect": cv2.BORDER_REFLECT,
    "extend": cv2.BORDER_REPLICATE,
    "wrap": cv2.BORDER_WRAP,
}

TEST_COMBOS = [
    ("blur_3x3", "reflect", "grayscale"),
    ("blur_3x3", "reflect", "rgb"),
    ("emboss_3x3", "reflect", "grayscale"),
    ("blur_3x3", "zero", "rgb"),
]


def get_kernel_array(kernel_name):
    kernel_list = KERNEL_FUNCS[kernel_name]
    return np.array(kernel_list, dtype=np.float32)


def measure_time(func, *args, runs=NUM_RUNS, warmup=WARMUP):
    """Измеряет среднее время выполнения функции (в миллисекундах)."""
    for _ in range(warmup):
        func(*args)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return np.mean(times), np.std(times)


def my_grayscale_convolution(image, kernel_name, edge_mode):
    kernel = get_kernel_array(kernel_name)
    return apply_convolution(image, kernel, edge_mode)


def my_rgb_convolution(image, kernel_name, edge_mode):
    kernel = get_kernel_array(kernel_name)
    return apply_convolution_rgb(image, kernel, edge_mode)


def opencv_grayscale_convolution(image, kernel_name, edge_mode):
    kernel = get_kernel_array(kernel_name)
    border = OPENCV_BORDER[edge_mode]
    if border == cv2.BORDER_WRAP:
        raise ValueError("OpenCV filter2D does not support BORDER_WRAP")
    return cv2.filter2D(image, -1, kernel, borderType=border)


def opencv_rgb_convolution(image, kernel_name, edge_mode):
    kernel = get_kernel_array(kernel_name)
    border = OPENCV_BORDER[edge_mode]
    if border == cv2.BORDER_WRAP:
        raise ValueError("OpenCV filter2D does not support BORDER_WRAP")
    return cv2.filter2D(image, -1, kernel, borderType=border)


def run_benchmark(output_dir="benchmark_results"):
    """
    Запускает бенчмарк и сохраняет результаты в указанную папку.
    Возвращает список результатов.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_gb": psutil.virtual_memory().total / (1024**3),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }
    with open(out_dir / "system_info.json", "w") as f:
        json.dump(system_info, f, indent=2)

    results = []
    np.random.seed(42)

    for size in SIZES:
        print(f"\nTesting size: {size}x{size}")
        gray_img = np.random.rand(size, size).astype(np.float32) * 255
        rgb_img = np.random.rand(size, size, 3).astype(np.float32) * 255

        for kernel_name, edge_mode, img_type in TEST_COMBOS:
            if edge_mode == "wrap":
                continue
            print(f"Testing: {kernel_name} | {edge_mode} | {img_type}")

            if img_type == "grayscale":

                def my_func():
                    return my_grayscale_convolution(gray_img, kernel_name, edge_mode)

                def cv_func():
                    return opencv_grayscale_convolution(gray_img, kernel_name, edge_mode)

            else:

                def my_func():
                    return my_rgb_convolution(rgb_img, kernel_name, edge_mode)

                def cv_func():
                    return opencv_rgb_convolution(rgb_img, kernel_name, edge_mode)

            try:
                my_mean, my_std = measure_time(my_func)
                cv_mean, cv_std = measure_time(cv_func)
                speedup = my_mean / cv_mean
                results.append(
                    {
                        "size": size,
                        "kernel": kernel_name,
                        "edge_mode": edge_mode,
                        "image_type": img_type,
                        "my_mean_ms": my_mean,
                        "my_std_ms": my_std,
                        "cv_mean_ms": cv_mean,
                        "cv_std_ms": cv_std,
                        "speedup": speedup,
                    }
                )
                print(
                    f"My: {my_mean:.2f} ± {my_std:.2f} ms, "
                    f"OpenCV: {cv_mean:.2f} ± {cv_std:.2f} ms, "
                    f"Speedup: {speedup:.1f}x"
                )
            except Exception as e:
                print(f"ERROR: {e}")

    if results:
        with open(out_dir / "benchmark_res.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        with open(out_dir / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nРезультаты сохранены в {out_dir}")
    else:
        print("Нет результатов для сохранения")
    return results


if __name__ == "__main__":
    run_benchmark()
