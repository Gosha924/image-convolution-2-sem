# image-convolution-2-sem

- **Инструмент для свёртки изображений с различными ядрами и методами обработки краёв. Поддерживает работу как с цветными, так и с полутоновыми изображениями.**

## Возможности

- **Ядра свёртки** размером 3×3 и 5×5:
  - `blur` – усреднённое размытие 
  - `sharpness` – повышение резкости
  - `emboss` – тиснение
  - `gaussian_blur` – размытие по Гауссу 3×3
  - `highlighting_vertical_borders` – выделение вертикальных границ
  - `highlighting_horizontal_borders` – выделение горизонтальных границ
  - `box_blur_5x5` – усреднённое размытие 5×5
  - `gaussian_blur_5x5` – размытие по Гауссу 5×5

- **Методы обработки краёв**:
  - `zero` – пиксели за границей считаются чёрными (игнорируются)
  - `reflection` – зеркальное отражение края
  - `extend` – дублирование крайних пикселей (расширение)
  - `wrap` – циклическое замыкание (изображение как бесконечная плитка)


# Аргументы командной строки

 - -o, --output	Путь для сохранения результата. Если не указан, изображение не сохраняется.
 - -k, --kernel	Имя ядра из списка выше. По умолчанию blur.
 - -e, --edge	Метод обработки краёв (zero, reflection, extend, wrap). По умолчанию zero.
 - -c, --color	Применить свёртку к каждому цветовому каналу RGB. Без флага изображение преобразуется в оттенки серого.

# 🚀 Запуск
```bash
    python -m src.main image1.png -o output2.png -k emboss
    python -m src.main image1.png -o output1.png -e zero
    python -m src.main image1.png -o output3.png -k gaussian_blur -c
    python -m src.main image1.png -o output4.png -k highlighting_horizontal_borders -c
    
```

# Тесты
```bash
  pytest tests/test_convolution.py -v
```

# Запуск бенчмарков
```bash
  python -m benchmark.run-benchmark
  python -m benchmark.visualization
```

# Лицензия изображений

#### Все тестовые изображения взяты с сайта https://www.rupixel.ru/. Согласно информации на сайте, все фотографии являются общественным достоянием (Public Domain).