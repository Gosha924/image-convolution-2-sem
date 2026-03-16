def reflection_metod(pixel_x, pixel_y, width, height):
    """
    обработка края  при свёртке с помощью отражения
    """
    if pixel_x < 0:
        pixel_x = -pixel_x
    elif pixel_x >= width:
        pixel_x = width - (pixel_x - width) - 1
    if pixel_y < 0:
        pixel_y = -pixel_y
    elif pixel_y >= height:
        pixel_y = height - (pixel_y - height) - 1
    return (pixel_x, pixel_y)