from pathlib import Path

def get_image_path_for_read(image_name) -> str:
    """
    Возвращает полный путь к изображению в папке images
    """
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    image_path = project_root / "images" / image_name
    return image_path

def get_image_path_for_save(image_name) -> str:
    """
    Возвращает полный путь к изображению в папке results
    """
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    image_path = project_root / "results" / image_name
    return image_path

