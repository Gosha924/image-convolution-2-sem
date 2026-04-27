def reflection_method(pixel_x: int, pixel_y: int, width: int, height: int) -> tuple[int, int]:
    """
    обработка края  при свёртке с помощью отражения
    """
    if pixel_x < 0:
        pixel_x = -pixel_x - 1
    elif pixel_x >= width:
        pixel_x = 2 * width - pixel_x - 1
    if pixel_y < 0:
        pixel_y = -pixel_y - 1
    elif pixel_y >= height:
        pixel_y = 2 * height - pixel_y - 1
    return pixel_x, pixel_y


def zero_method(pixel_x: int, pixel_y: int, width: int, height: int):
    if 0 <= pixel_x < width and 0 <= pixel_y < height:
        return pixel_x, pixel_y
    return None


def extend_method(pixel_x: int, pixel_y: int, width: int, height: int) -> tuple[int, int]:
    """
    Расширение края — координаты прижимаются к границе
    """
    pixel_x = max(0, min(pixel_x, width - 1))
    pixel_y = max(0, min(pixel_y, height - 1))
    return pixel_x, pixel_y


def wrap_method(pixel_x: int, pixel_y: int, width: int, height: int) -> tuple[int, int]:
    """
    Циклическое замыкание
    """
    pixel_x = pixel_x % width
    pixel_y = pixel_y % height
    return pixel_x, pixel_y
