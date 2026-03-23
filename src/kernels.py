def blur_kernel() -> list[list[float]]:
    return [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]


def sharpness_kernel():
    return [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]


def emboss_kernel() -> list[list[int]]:
    """
    нужно прибавлять к результату 128 чтобы было видно на сером фоне
    """
    return [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]
