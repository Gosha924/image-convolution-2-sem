import numpy as np

from src.main import apply_convolution


def test_identity_kernel():
    """Ядро с единицей в центре не должно менять изображение"""
    image = np.random.rand(10, 10).astype(np.float64) * 255
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
    result = apply_convolution(image, kernel, edge_mode="reflect")
    assert np.allclose(result, image, atol=1e-6)


def test_blur_on_constant():
    """Усредняющее ядро на постоянном изображении не меняет яркость"""
    image = np.full((10, 10), 127.0, dtype=np.float64)
    kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    result = apply_convolution(image, kernel, edge_mode="reflect")
    assert np.allclose(result, image, atol=1e-6)


def test_zero_padding_edge():
    """Проверка, что при edge_mode='zero' края не искажаются"""
    image = np.ones((5, 5), dtype=np.float64) * 100
    kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    result = apply_convolution(image, kernel, edge_mode="zero")
    # Центральный пиксель должен остаться 100
    assert result[2, 2] == 100.0
    # Угловой пиксель должен быть меньше из-за нулей
    assert result[0, 0] < 100.0
