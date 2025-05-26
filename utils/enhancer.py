# =============================================================================
# Импорты и конфигурация
# =============================================================================
import replicate
import requests
import zipfile
import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw, ImageChops
import io
import numpy as np
import uuid
from insightface.app import FaceAnalysis
import onnxruntime

# =============================================================================
# Инициализация моделей
# =============================================================================
def ensure_insightface_model():
    """Загрузка и подготовка модели InsightFace для анализа лиц."""
    model_dir = "models/buffalo_l"
    if not os.path.exists(model_dir):
        print("Скачиваю модель buffalo_l...")

        os.makedirs("models", exist_ok=True)
        url = "https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l.zip"
        headers = {"User-Agent": "Mozilla/5.0"}
        zip_path = "models/buffalo_l.zip"

        response = requests.get(url, headers=headers, allow_redirects=True)
        if not response.ok or b"PK" not in response.content[:4]:
            raise Exception("Ошибка при скачивании: файл не является zip-архивом.")

        with open(zip_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("models/")
        os.remove(zip_path)
        print("Модель buffalo_l загружена.")

# Инициализация анализатора лиц
ensure_insightface_model()
face_analyzer = FaceAnalysis(name="buffalo_l", root="models", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0)

# =============================================================================
# Базовые функции анализа изображения
# =============================================================================
def has_face(image_path: str) -> bool:
    """Проверка наличия лица на фотографии."""
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    faces = face_analyzer.get(img_np)
    print(f"Обнаружено лиц: {len(faces)}")
    return len(faces) > 0

def get_face_brightness_live(image: Image.Image) -> float:
    """Определение яркости лица на изображении."""
    img_np = np.array(image)
    faces = face_analyzer.get(img_np)
    if not faces:
        return np.mean(np.array(image.convert("L")))
    face = faces[0]
    x1, y1, x2, y2 = map(int, face.bbox)
    face_crop = image.crop((x1, y1, x2, y2)).convert("L")
    return np.mean(np.array(face_crop))

def is_dark_photo(image: Image.Image) -> bool:
    """Определение темного изображения."""
    gray = image.convert("L")
    avg_brightness = np.array(gray).mean()
    return avg_brightness < 50

def analyze_skin_tone(image: Image.Image, face_data) -> str:
    """Анализ тона кожи на фотографии."""
    if not face_data:
        return "unknown"

    x1, y1, x2, y2 = map(int, face_data.bbox)
    face_crop = image.crop((x1, y1, x2, y2))
    face_np = np.array(face_crop).astype(np.float32)

    r_mean = np.mean(face_np[:, :, 0])
    g_mean = np.mean(face_np[:, :, 1])
    b_mean = np.mean(face_np[:, :, 2])

    red_yellow_ratio = r_mean / (g_mean + 1)
    blue_ratio = b_mean / ((r_mean + g_mean) / 2 + 1)

    if r_mean < 100 and g_mean < 100 and b_mean < 100:
        return "pale"
    elif red_yellow_ratio > 1.25:
        return "red"
    elif r_mean > g_mean and g_mean > b_mean:
        return "yellow"
    elif blue_ratio > 1.1:
        return "cold"
    else:
        return "balanced"

def classify_scene(image: Image.Image) -> str:
    """Классификация сцены на фотографии."""
    gray = image.convert("L")
    brightness = np.mean(np.array(gray))
    r, g, b = image.split()
    red_avg = np.mean(np.array(r))
    blue_avg = np.mean(np.array(b))

    if brightness < 55:
        return "night"
    elif blue_avg > red_avg + 20:
        return "cold_white_light"
    elif brightness > 200:
        return "overexposed"
    else:
        return "day"

# =============================================================================
# Функции улучшения изображения
# =============================================================================
def enhance_eyes_and_lips(image: Image.Image, face_data) -> Image.Image:
    """Улучшение глаз и губ на фотографии."""
    if not face_data or not hasattr(face_data, "landmark_2d_106"):
        return image

    img = image.copy()
    landmarks = face_data.landmark_2d_106

    def apply_soft_overlay(points, enhance_fn):
        xs = [int(p[0]) for p in points]
        ys = [int(p[1]) for p in points]
        x1, x2 = max(min(xs), 0), min(max(xs), img.width)
        y1, y2 = max(min(ys), 0), min(max(ys), img.height)
        box = (x1, y1, x2, y2)

        region = img.crop(box)
        enhanced = enhance_fn(region)

        mask = Image.new("L", region.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, region.size[0], region.size[1]), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=4))

        base = img.crop(box)
        blended = Image.composite(enhanced, base, mask)
        img.paste(blended, box)

    apply_soft_overlay(landmarks[70:89], lambda r: ImageEnhance.Color(r).enhance(1.25))
    return img

def apply_subject_lighting(image: Image.Image) -> Image.Image:
    """Применение освещения для объекта."""
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.08)
    warm_overlay = Image.new("RGB", image.size, (255, 230, 200))
    image = Image.blend(image, warm_overlay, 0.03)
    return image

def conditional_brightness(image: Image.Image) -> Image.Image:
    """Адаптивная регулировка яркости."""
    avg_brightness = get_face_brightness_live(image)
    if avg_brightness > 130:
        brightness_factor = 1.00
    elif avg_brightness > 120:
        brightness_factor = 1.08
    else:
        brightness_factor = np.clip(1.4 - (avg_brightness - 80) * 0.00375, 1.1, 1.4)
    return ImageEnhance.Brightness(image).enhance(brightness_factor)

def lighten_skin_and_hair_only(image: Image.Image, face_data) -> Image.Image:
    """Мягкое осветление кожи и волос без размытия."""
    img = image.copy()

    x1, y1, x2, y2 = map(int, face_data.bbox)
    width, height = img.size

    # Область вокруг лица
    margin_x = int((x2 - x1) * 0.6)
    margin_y = int((y2 - y1) * 0.9)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    ex1 = max(center_x - margin_x, 0)
    ey1 = max(center_y - margin_y, 0)
    ex2 = min(center_x + margin_x, width)
    ey2 = min(center_y + margin_y, height)

    region = img.crop((ex1, ey1, ex2, ey2))

    # Осветление без размытия
    brightened = ImageEnhance.Brightness(region).enhance(1.08)
    overlay = Image.new("RGB", region.size, (250, 235, 220))
    blended = Image.blend(brightened, overlay, 0.03)

    # Плавная маска перехода
    mask = Image.new("L", region.size, 0)
    draw = ImageDraw.Draw(mask)
    
    for y in range(region.size[1]):
        for x in range(region.size[0]):
            dx = (x - region.size[0] / 2) / (region.size[0] / 2)
            dy = (y - region.size[1] / 2) / (region.size[1] / 2)
            distance = (dx ** 2 + dy ** 2) ** 0.5
            alpha = max(0, min(255, int(255 * (1 - distance))))
            mask.putpixel((x, y), alpha)

    # Минимальное размытие только для маски перехода
    mask = mask.filter(ImageFilter.GaussianBlur(radius=10))

    base = img.crop((ex1, ey1, ex2, ey2))
    final_region = Image.composite(blended, base, mask)
    img.paste(final_region, (ex1, ey1))

    return img

def enhance_face_lighting(image: Image.Image, face_data) -> Image.Image:
    """Улучшение освещения лица без размытия."""
    if not face_data:
        return image

    img = image.copy()
    x1, y1, x2, y2 = map(int, face_data.bbox)

    # Небольшое расширение области
    padding_x = int((x2 - x1) * 0.1)
    padding_y = int((y2 - y1) * 0.1)
    
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(img.width, x2 + padding_x)
    y2 = min(img.height, y2 + padding_y)

    face_crop = img.crop((x1, y1, x2, y2))
    # Осветление без размытия
    bright_face = ImageEnhance.Brightness(face_crop).enhance(1.08)
    warm_overlay = Image.new("RGB", bright_face.size, (255, 230, 200))
    enhanced = Image.blend(bright_face, warm_overlay, 0.04)

    # Плавная маска перехода
    mask = Image.new("L", (x2 - x1, y2 - y1), 0)
    draw = ImageDraw.Draw(mask)
    
    for y in range(y2 - y1):
        for x in range(x2 - x1):
            dx = (x - (x2 - x1) / 2) / ((x2 - x1) / 2)
            dy = (y - (y2 - y1) / 2) / ((y2 - y1) / 2)
            distance = (dx ** 2 + dy ** 2) ** 0.5
            alpha = max(0, min(255, int(255 * (1 - distance))))
            mask.putpixel((x, y), alpha)

    # Минимальное размытие только для маски
    mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
    
    base = img.crop((x1, y1, x2, y2))
    final_region = Image.composite(enhanced, base, mask)
    img.paste(final_region, (x1, y1))

    return img

def apply_soft_filter(image: Image.Image, intensity: str = "normal") -> Image.Image:
    """Применение мягкого фильтра."""
    base = image.copy()

    if intensity == "strong":
        blur_radius = 3.2
        brightness_factor = 1.10
        contrast_factor = 1.05
        alpha = 0.35
    else:
        blur_radius = 2.2
        brightness_factor = 1.07
        contrast_factor = 1.03
        alpha = 0.25

    blurred = base.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    brightened = ImageEnhance.Brightness(blurred).enhance(brightness_factor)
    contrast = ImageEnhance.Contrast(brightened).enhance(contrast_factor)
    result = Image.blend(base, contrast, alpha=alpha)

    return result
    
def adjust_by_skin_tone(image: Image.Image, tone: str) -> Image.Image:
    """Коррекция изображения на основе тона кожи."""
    img = image.copy()

    if tone == "pale":
        img = ImageEnhance.Color(img).enhance(1.15)
        overlay = Image.new("RGB", img.size, (70, 50, 30))
        img = Image.blend(img, overlay, 0.06)
    elif tone == "red":
        r, g, b = img.split()
        r = r.point(lambda i: i * 0.94)
        img = Image.merge("RGB", (r, g, b))
    elif tone == "yellow":
        r, g, b = img.split()
        g = g.point(lambda i: i * 0.92)
        img = Image.merge("RGB", (r, g, b))
    elif tone == "cold":
        overlay = Image.new("RGB", img.size, (80, 60, 40))
        img = Image.blend(img, overlay, 0.05)

    return img
    
def apply_background_blur(image: Image.Image, face_data) -> Image.Image:
    """Размытие фона на фотографии."""
    if not face_data:
        return image

    width, height = image.size
    x1, y1, x2, y2 = map(int, face_data.bbox)

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    ellipse_width = int((x2 - x1) * 2.0)
    ellipse_height = int((y2 - y1) * 3.5)

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([
        max(center_x - ellipse_width // 2, 0),
        max(center_y - ellipse_height // 2, 0),
        min(center_x + ellipse_width // 2, width),
        min(center_y + ellipse_height // 2, height)
    ], fill=255)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=30))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=1.8))
    return Image.composite(image, blurred, ImageOps.invert(mask))

def apply_scenario(image: Image.Image, face_data, scene_type: str) -> Image.Image:
    """Применение эффектов в зависимости от сценария."""
    img = image.copy()

    face_mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(face_mask)
    if face_data:
        x1, y1, x2, y2 = map(int, face_data.bbox)
        draw.rectangle([x1, y1, x2, y2], fill=255)

    if scene_type == "night":
        img = apply_subject_lighting(img)
    elif scene_type == "cold_white_light":
        warm_overlay = Image.new("RGB", img.size, (30, 20, 10))
        img = Image.blend(img, warm_overlay, alpha=0.08)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=1.3))
        img.paste(blurred, mask=face_mask)
    elif scene_type == "overexposed":
        img = ImageEnhance.Contrast(img).enhance(0.95)
        img = ImageEnhance.Brightness(img).enhance(0.95)

    if face_data:
        x1, y1, x2, y2 = map(int, face_data.bbox)
        face_region = img.crop((x1, y1, x2, y2))
        bright_face = ImageEnhance.Brightness(face_region).enhance(1.10)
        img.paste(bright_face, (x1, y1))

    return img

def apply_final_polish(image: Image.Image) -> Image.Image:
    """Финальная обработка изображения."""
    image = conditional_brightness(image)
    
    avg_brightness = np.array(image.convert("L")).mean()
    
    if avg_brightness > 100:
        image = ImageEnhance.Contrast(image).enhance(1.10)
    else:
        image = ImageEnhance.Contrast(image).enhance(1.02)
    image = ImageEnhance.Color(image).enhance(1.10)

    avg_brightness = np.array(image.convert("L")).mean()
    if avg_brightness > 130:
        sharpness = 1.40
    elif avg_brightness > 100:
        sharpness = 1.25
    else:
        sharpness = 1.05
    image = ImageEnhance.Sharpness(image).enhance(sharpness)

    warm_overlay = Image.new("RGB", image.size, (255, 245, 225))
    image = Image.blend(image, warm_overlay, 0.03)
    image = ImageEnhance.Brightness(image).enhance(1.06)

    image = ImageEnhance.Brightness(image).enhance(1.06)
    r, g, b = image.split()
    b = b.point(lambda i: i * 0.97)
    image = Image.merge("RGB", (r, g, b))

    image = ImageEnhance.Brightness(image).enhance(1.03)
    warm_overlay = Image.new("RGB", image.size, (255, 225, 190))
    image = Image.blend(image, warm_overlay, 0.015)

    return image

def detect_light_source_color(image: Image.Image) -> tuple:
    """Определение цвета основного источника света."""
    img_np = np.array(image)
    # Находим самые яркие области
    gray = np.mean(img_np, axis=2)
    bright_mask = gray > np.percentile(gray, 90)  # Еще больше снижаем порог
    if not np.any(bright_mask):
        return (1.0, 1.0, 1.0)
    
    # Анализируем цвет ярких областей
    bright_colors = img_np[bright_mask]
    avg_color = np.mean(bright_colors, axis=0) / 255.0
    
    # Более агрессивное определение желтого света
    r, g, b = avg_color
    is_yellow = (r > 0.5 and g > 0.5 and b < 0.45) or (r/b > 1.4 and g/b > 1.4)
    if is_yellow:
        # Усиливаем компенсацию для желтого света
        return (r * 0.85, g * 0.80, b * 1.3)
    
    return tuple(avg_color)

def compensate_light_color(image: Image.Image, face_data) -> Image.Image:
    """Компенсация цветового оттенка освещения."""
    light_color = detect_light_source_color(image)
    
    # Если освещение имеет сильный цветовой оттенок
    max_channel = max(light_color)
    if max_channel > 0:
        r_ratio, g_ratio, b_ratio = [max_channel/c if c > 0.2 else 1.0 for c in light_color]
        
        # Более агрессивная коррекция для желтого света
        if r_ratio > 1.1 and g_ratio > 1.1 and b_ratio < 0.9:
            r_ratio = np.clip(r_ratio * 0.85, 0.7, 1.1)
            g_ratio = np.clip(g_ratio * 0.85, 0.7, 1.1)
            b_ratio = np.clip(b_ratio * 1.2, 0.9, 1.3)
        else:
            # Стандартная коррекция для других случаев
            r_ratio = np.clip(r_ratio, 0.85, 1.15)
            g_ratio = np.clip(g_ratio, 0.85, 1.15)
            b_ratio = np.clip(b_ratio, 0.85, 1.15)
        
        img = image.copy()
        r, g, b = img.split()
        
        # Применяем коррекцию
        r = r.point(lambda x: min(255, int(x * r_ratio)))
        g = g.point(lambda x: min(255, int(x * g_ratio)))
        b = b.point(lambda x: min(255, int(x * b_ratio)))
        
        return Image.merge('RGB', (r, g, b))
    return image

def apply_advanced_noise_reduction(image: Image.Image, strength: str = 'normal') -> Image.Image:
    """Продвинутое шумоподавление с сохранением деталей."""
    img = image.copy()
    
    # Параметры в зависимости от силы шумоподавления
    if strength == 'strong':
        # Начальное сильное шумоподавление
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        # Создаем несколько размытых версий
        blur1 = img.filter(ImageFilter.GaussianBlur(radius=1.2))
        blur2 = img.filter(ImageFilter.GaussianBlur(radius=0.6))
        
        # Находим края
        edge_mask = img.filter(ImageFilter.FIND_EDGES)
        edge_mask = edge_mask.convert('L')
        edge_mask = edge_mask.point(lambda x: 255 if x > 15 else 0)
        edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        # Создаем маску деталей
        detail_mask = img.filter(ImageFilter.DETAIL)
        detail_mask = detail_mask.convert('L')
        detail_mask = detail_mask.point(lambda x: 255 if x > 10 else 0)
        
        # Комбинируем маски
        final_mask = ImageChops.lighter(edge_mask, detail_mask)
        final_mask = final_mask.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        # Смешиваем версии
        result = Image.composite(img, blur1, final_mask)
        result = Image.composite(result, blur2, edge_mask)
    else:
        # Стандартное шумоподавление
        img = img.filter(ImageFilter.MedianFilter(size=3))
        blurred = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        edge_mask = img.filter(ImageFilter.FIND_EDGES)
        edge_mask = edge_mask.convert('L')
        edge_mask = edge_mask.point(lambda x: 255 if x > 20 else 0)
        edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        result = Image.composite(img, blurred, edge_mask)
    
    return result

def is_headlight_orange(color):
    """Определение оранжевого оттенка от фар."""
    r, g, b = color
    
    # Конвертируем в HSV для более точного определения оттенка
    r, g, b = r/255.0, g/255.0, b/255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    
    if max_val == 0:
        return False
        
    diff = max_val - min_val
    
    if max_val == r:
        hue = 60 * ((g - b) / diff % 6)
    elif max_val == g:
        hue = 60 * ((b - r) / diff + 2)
    else:
        hue = 60 * ((r - g) / diff + 4)
        
    if hue < 0:
        hue += 360
        
    saturation = 0 if max_val == 0 else diff / max_val
    value = max_val
    
    # Проверяем соответствие оранжевому от фар (#ff812f)
    # HSV: 24°, 82%, 100%
    return (
        (15 <= hue <= 35) and  # Оранжевый диапазон
        (saturation >= 0.7) and  # Высокая насыщенность
        (value >= 0.8)  # Высокая яркость
    )

def normalize_skin_tone(face_region: Image.Image) -> Image.Image:
    """Нормализация цвета кожи к естественному теплому белому."""
    # Анализируем текущий цвет кожи
    img_np = np.array(face_region)
    skin_mask = (img_np[:,:,0] > 60) & (img_np[:,:,1] > 60) & (img_np[:,:,2] > 60)
    if not np.any(skin_mask):
        return face_region
    
    skin_pixels = img_np[skin_mask]
    avg_color = np.mean(skin_pixels, axis=0)
    
    # Проверяем на оранжевый оттенок от фар
    is_headlight = is_headlight_orange(avg_color)
    
    # Определяем другие неестественные оттенки
    r, g, b = avg_color
    is_orange = r/b > 1.4 and g/b > 1.2
    is_yellow = r/b > 1.3 and g/b > 1.3
    
    if is_headlight or is_orange or is_yellow:
        # Для неестественных оттенков просто возвращаем оригинал
        return face_region
    
    # Для естественных оттенков можем немного улучшить
    enhanced = face_region.copy()
    enhanced = ImageEnhance.Color(enhanced).enhance(1.05)  # Легкое усиление цвета
    return enhanced

def create_person_mask(image: Image.Image, face_data) -> Image.Image:
    """Создание маски для области человека (голова, тело, волосы)."""
    if not face_data:
        return None
        
    width, height = image.size
    x1, y1, x2, y2 = map(int, face_data.bbox)
    
    # Расширяем область для захвата волос и части тела
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Расширение вверх для волос
    top_expand = int(face_height * 0.7)
    # Расширение вниз для шеи и части тела
    bottom_expand = int(face_height * 1.2)
    # Расширение по бокам
    side_expand = int(face_width * 0.3)
    
    # Вычисляем координаты расширенной области
    ex1 = max(x1 - side_expand, 0)
    ey1 = max(y1 - top_expand, 0)
    ex2 = min(x2 + side_expand, width)
    ey2 = min(y2 + bottom_expand, height)
    
    # Создаем маску
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Рисуем основную область
    draw.ellipse([ex1, ey1, ex2, ey2], fill=200)
    
    # Добавляем дополнительную область для тела
    body_points = [
        (ex1, ey2),
        (ex2, ey2),
        (ex2 + face_width//2, min(height, ey2 + face_height)),
        (ex1 - face_width//2, min(height, ey2 + face_height))
    ]
    draw.polygon(body_points, fill=180)
    
    # Сглаживаем края маски
    mask = mask.filter(ImageFilter.GaussianBlur(radius=face_width//8))
    
    return mask

def apply_premium_noise_reduction(image: Image.Image, strength: str = 'normal') -> Image.Image:
    """Премиум шумоподавление с сохранением деталей и текстур."""
    img = image.copy()
    
    # Параметры в зависимости от силы шумоподавления
    if strength == 'strong':
        # Многопроходное шумоподавление для ночных фото
        
        # Первый проход - сильное размытие для удаления шума
        blur1 = img.filter(ImageFilter.GaussianBlur(radius=2.0))
        
        # Второй проход - умное размытие с сохранением краев
        blur2 = img.filter(ImageFilter.MedianFilter(size=3))
        blur2 = blur2.filter(ImageFilter.GaussianBlur(radius=1.0))
        
        # Создаем маску деталей
        detail_mask = img.filter(ImageFilter.DETAIL)
        detail_mask = detail_mask.convert('L')
        detail_mask = detail_mask.point(lambda x: min(255, int(x * 1.5)))
        detail_mask = detail_mask.filter(ImageFilter.GaussianBlur(radius=1.0))
        
        # Создаем маску краев
        edge_mask = img.filter(ImageFilter.FIND_EDGES)
        edge_mask = edge_mask.convert('L')
        edge_mask = edge_mask.point(lambda x: 255 if x > 20 else x//2)
        edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(radius=1.0))
        
        # Комбинируем маски
        final_mask = ImageChops.lighter(edge_mask, detail_mask)
        final_mask = final_mask.filter(ImageFilter.GaussianBlur(radius=1.0))
        
        # Создаем версию с сохраненными деталями
        detail_preserved = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=150, threshold=3))
        
        # Смешиваем все версии
        result = Image.composite(detail_preserved, blur1, final_mask)
        result = Image.composite(result, blur2, edge_mask)
        
        # Финальное легкое размытие для сглаживания переходов
        result = result.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Восстанавливаем немного резкости для текстур
        result = result.filter(ImageFilter.UnsharpMask(radius=0.7, percent=50, threshold=3))
        
        return result
    else:
        # Стандартное шумоподавление для обычных фото
        return apply_advanced_noise_reduction(img, 'normal')

def detect_scene_type(image: Image.Image) -> str:
    """Определение типа сцены на фотографии."""
    img_np = np.array(image)
    
    # Анализируем общую яркость
    brightness = np.mean(img_np)
    
    # Анализируем цветовые каналы
    r, g, b = image.split()
    r_mean = np.mean(np.array(r))
    g_mean = np.mean(np.array(g))
    b_mean = np.mean(np.array(b))
    
    # Определяем цвет освещения
    light_color = detect_light_source_color(image)
    has_colored_light = max(light_color) / (min(light_color) + 0.01) > 1.3
    
    # Проверяем наличие цветного освещения (клубное)
    is_club = (
        (b_mean > (r_mean + g_mean) / 2 + 15) or  # Синее освещение
        (r_mean > (g_mean + b_mean) / 2 + 15) or  # Красное освещение
        (brightness < 90 and has_colored_light)    # Темно с цветным светом
    )
    
    # Проверяем равномерность освещения
    std_dev = np.std(img_np)
    is_uniform = std_dev < 50
    
    if is_club:
        return "club"
    elif brightness > 160 and is_uniform:
        return "daylight"
    else:
        return "evening"

def enhance_person_region(image: Image.Image, face_data, scene_type: str = "day") -> Image.Image:
    """Улучшение только области человека."""
    if not face_data:
        return image
        
    img = image.copy()
    
    # Общее осветление для всего изображения
    enhanced = img.copy()
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.08)
    
    # Проверяем цвет кожи
    x1, y1, x2, y2 = map(int, face_data.bbox)
    face_region = enhanced.crop((x1, y1, x2, y2))
    img_np = np.array(face_region)
    skin_pixels = img_np[(img_np[:,:,0] > 60) & (img_np[:,:,1] > 60) & (img_np[:,:,2] > 60)]
    if len(skin_pixels) > 0:
        avg_color = np.mean(skin_pixels, axis=0)
        is_unnatural = (
            is_headlight_orange(avg_color) or 
            avg_color[0]/avg_color[2] > 1.4 or 
            avg_color[1]/avg_color[2] > 1.3
        )
    else:
        is_unnatural = False

    if scene_type == "club":
        # Специальная обработка для клубных фото
        enhanced = apply_premium_noise_reduction(enhanced, 'strong')
        
        if not is_unnatural:
            enhanced = ImageEnhance.Brightness(enhanced).enhance(1.12)
            enhanced = ImageEnhance.Contrast(enhanced).enhance(1.06)
        
    elif scene_type == "day":
        # Для дневных фото
        if not is_unnatural:
            enhanced = ImageEnhance.Brightness(enhanced).enhance(1.10)
            enhanced = ImageEnhance.Contrast(enhanced).enhance(1.05)
            
            # Добавляем теплый оттенок
            warm_overlay = Image.new("RGB", enhanced.size, (255, 240, 230))
            enhanced = Image.blend(enhanced, warm_overlay, 0.05)
        
    else:
        # Для вечерних фото
        enhanced = apply_premium_noise_reduction(enhanced, 'strong')
        
        if not is_unnatural:
            enhanced = ImageEnhance.Brightness(enhanced).enhance(1.12)
            enhanced = ImageEnhance.Contrast(enhanced).enhance(1.06)
            
            # Добавляем теплый оттенок
            warm_overlay = Image.new("RGB", enhanced.size, (255, 240, 230))
            enhanced = Image.blend(enhanced, warm_overlay, 0.04)
    
    # Нормализация цвета кожи
    face_region = enhanced.crop((x1, y1, x2, y2))
    face_region = normalize_skin_tone(face_region)
    enhanced.paste(face_region, (x1, y1))
    
    return enhanced

def create_lighting_mask(size, x1, y1, x2, y2):
    """Создание маски освещения для лица и тела."""
    light_mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(light_mask)
    
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Центр лица
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Радиус для градиентного освещения
    main_radius = max(face_width, face_height) * 1.5
    
    # Основное освещение лица и тела
    for r in range(int(main_radius * 2)):
        opacity = int(255 * (1 - (r / (main_radius * 2)) ** 1.5))
        if opacity > 0:
            draw.ellipse([
                center_x - r, center_y - r * 1.2,
                center_x + r, center_y + r * 1.5
            ], fill=opacity)
    
    # Дополнительное освещение для волос
    hair_y = y1 - face_height * 0.3
    hair_radius = face_width * 0.8
    for r in range(int(hair_radius * 2)):
        opacity = int(180 * (1 - (r / (hair_radius * 2)) ** 1.2))
        if opacity > 0:
            draw.ellipse([
                center_x - r, hair_y - r,
                center_x + r, hair_y + r * 1.5
            ], fill=opacity)
    
    # Размываем маску для плавности
    return light_mask.filter(ImageFilter.GaussianBlur(radius=face_width//6))

def apply_daylight_enhancement(image: Image.Image, face_data) -> Image.Image:
    """Улучшение фотографии при дневном освещении."""
    return enhance_person_region(image, face_data, "day")

def apply_evening_enhancement(image: Image.Image, face_data) -> Image.Image:
    """Улучшение фотографии при вечернем/искусственном освещении."""
    return enhance_person_region(image, face_data, "evening")

# =============================================================================
# Основные функции обработки
# =============================================================================
async def enhance_image(image_bytes: bytes, user_prompt: str = "") -> bytes:
    """Улучшение изображения с использованием IDNBeauty."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)

    temp_filename = f"{uuid.uuid4()}.jpg"
    image.save(temp_filename)

    if not has_face(temp_filename):
        os.remove(temp_filename)
        raise Exception("Лицо не обнаружено. Пожалуйста, загрузите чёткий портрет.")

    try:
        prompt = (
            "Natural face enhancement, preserve facial features and colors. "
            "Keep eyes 100% original, do not modify eyes shape or color. "
            "Do not change eye makeup, lashes or eyebrows. "
            "Keep exact eye direction and gaze. "
            "Do not modify eye corners or eye size. "
            "Enhance skin texture moderately. "
            "Keep all facial features in original positions. "
            "Balanced enhancement only."
        )
        if user_prompt:
            prompt = user_prompt

        idnbeauty_result = replicate.run(
            "torrikabe-ai/idnbeauty:5f994656b3b88df2e21a3cf0a81371d66bd6ff45171f3e5618bb314bdc8b64b1",
            input={
                "image": open(temp_filename, "rb"),
                "prompt": prompt,
                "model": "dev",
                "guidance_scale": 0.4,     # Средний уровень влияния
                "prompt_strength": 0.08,    # Средняя сила промпта
                "num_inference_steps": 22,  # Оптимальное количество шагов
                "output_format": "png",
                "output_quality": 90,
                "go_fast": True,
                "lora_scale": 0.5,         # Средний уровень влияния LoRA
                "extra_lora_scale": 0.07    # Средний уровень дополнительного влияния
            }
        )

        response = requests.get(str(idnbeauty_result[0]))
        image_idn = Image.open(io.BytesIO(response.content)).convert("RGB")

        img_np = np.array(image_idn)
        faces = face_analyzer.get(img_np)
        face = faces[0] if faces else None
        scene_type = classify_scene(image_idn)
        skin_tone = analyze_skin_tone(image_idn, face)

        # Сначала корректируем цвет кожи
        if face:
            x1, y1, x2, y2 = map(int, face.bbox)
            face_region = image_idn.crop((x1, y1, x2, y2))
            face_region = normalize_skin_tone(face_region)
            image_idn.paste(face_region, (x1, y1))

        # Затем применяем остальные улучшения
        final_image = enhance_person_region(image_idn, face, "day" if scene_type != "night" else "evening")

        final_bytes = io.BytesIO()
        final_image.save(final_bytes, format="JPEG", quality=99, subsampling=0)
        final_bytes.seek(0)

        return final_bytes.read()

    except Exception as e:
        os.remove(temp_filename)
        raise Exception(f"Ошибка обработки IDNBeauty или постобработки: {e}")

async def enhance_image_remini(image_bytes: bytes) -> bytes:
    """Улучшение изображения с определением сценария."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)

    temp_filename = f"{uuid.uuid4()}.jpg"
    image.save(temp_filename)

    if not has_face(temp_filename):
        os.remove(temp_filename)
        raise Exception("Лицо не обнаружено. Пожалуйста, загрузите чёткий портрет.")

    try:
        # Определяем тип сцены
        scene_type = detect_scene_type(image)
        
        # Анализ лиц
        img_np = np.array(image)
        faces = face_analyzer.get(img_np)
        face = faces[0] if faces else None

        # Применяем обработку в зависимости от сценария
        if face:
            if scene_type == "club":
                enhanced_image = apply_club_photo_enhancement(image, face)
            elif scene_type == "evening":
                enhanced_image = apply_evening_enhancement(image, face)
            else:
                enhanced_image = apply_daylight_enhancement(image, face)

        # Сохраняем результат
        final_bytes = io.BytesIO()
        enhanced_image.save(final_bytes, format="JPEG", quality=95, subsampling=0)
        final_bytes.seek(0)

        os.remove(temp_filename)
        return final_bytes.read()

    except Exception as e:
        os.remove(temp_filename)
        raise Exception(f"Ошибка при обработке фотографии: {e}")