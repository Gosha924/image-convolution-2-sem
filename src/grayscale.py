from PIL import Image
from numpy import array, uint8, zeros, float64, clip
from main import get_image_path_for_save, get_image_path_for_read
import edge_processing
import kernels

image = array(Image.open(get_image_path_for_read("image1.jpg")))

grayscale = 0.3 * image[:, :, 0] + 0.6 * image[:, :, 1] + 0.1 * image[:, :, 2]

kernel = kernels.blur_kernel()
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
        "итерируемся по ядру свертки"
        for ky in range(kernel_height):
            for kx in range(kernel_height):
                pixel_x = x + kx - kernel_half
                pixel_y = y + ky - kernel_half

                pixel_x, pixel_y = edge_processing.reflection_metod(pixel_x, pixel_y, width, height)

                pixel_value = grayscale[pixel_y][pixel_x]
                kernel_value = kernel[ky][kx]
                pixel_sum += pixel_value * kernel_value
        result[y][x] = pixel_sum

result = clip(result, 0, 255)
result = result.astype(uint8)
gray_image = Image.fromarray(result)
gray_image.save(get_image_path_for_save("output.jpg"))




