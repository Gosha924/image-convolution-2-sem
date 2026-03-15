from PIL import Image, ImageFilter
from pathlib import Path

def get_image_path(image_name) -> str:
    """
    Возвращает полный путь к изображению в папке images
    """
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    image_path = project_root / "images" / image_name
    return image_path

image = Image.open(get_image_path("image1.jpg"))
kernel = [0, -1, 0,
          -1, 5, -1,
          0, -1 , 0]
native_filter = image.filter(ImageFilter.CONTOUR)
my_first_filter = ImageFilter.Kernel((3, 3), kernel)
result = image.filter(my_first_filter)

native_filter.save("native_fiter.jpeg")
result.save("my_filter2.jpeg")

