blur_kernel: list[list[float]] = [
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
]

sharpness_kernel: list[list[float]] = [
    [0.0, -1.0, 0.0],
    [-1.0, 5.0, -1.0],
    [0.0, -1.0, 0.0],
]

emboss_kernel: list[list[float]] = [
    [-2.0, -1.0, 0.0],
    [-1.0, 1.0, 1.0],
    [0.0, 1.0, 2.0],
]

gaussian_blur: list[list[float]] = [
    [1 / 16, 2 / 16, 1 / 16],
    [2 / 16, 4 / 16, 2 / 16],
    [1 / 16, 2 / 16, 1 / 16],
]
