from PIL import Image
from numpy import array, uint8, zeros, float64, clip
from main import get_image_path_for_save, get_image_path_for_read

image = array(Image.open(get_image_path_for_read("image1.jpg")))

grayscale = 0.3 * image[:, :, 0] + 0.6 * image[:, :, 1] + 0.1 * image[:, :, 2]

kernel = [[1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]]
kernel = array(kernel, dtype=float64)
kernel_height = len(kernel)
kernel_half = kernel_height // 2
height = len(grayscale)
width = len(grayscale[1])
print(height, width)
result = zeros((height, width), dtype=float64)
for y in range(height):
    for x in range(width):
        pixel_sum = 0.0
        """ тут итерируемся по ядру свертки """
        for ky in range(kernel_height):
            for kx in range(kernel_height):
                pixel_x = x - kernel_half + kx
                pixel_y = y - kernel_half + ky

                if pixel_x < 0 or pixel_x >= width:
                    pixel_x = 0
                if pixel_y < 0 or pixel_y >= height:
                    pixel_y = 0


                pixel_value = grayscale[pixel_y][pixel_x]
                kernel_value = kernel[ky][kx]
                pixel_sum += pixel_value * kernel_value
        result[y, x] = pixel_sum
result = clip(result, 0, 255)
result = result.astype(uint8)
gray_image = Image.fromarray(result)
gray_image.save(get_image_path_for_save("output.jpeg"))




