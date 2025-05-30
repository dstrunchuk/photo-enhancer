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

def enhance_single_eye(image: Image.Image, points: list) -> Image.Image:
    """Улучшение одного глаза по точкам landmark с автоопределением центра."""
    img = image.copy()

    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]

    if not xs or not ys:
        return img

    # Центр и размеры
    cx = sum(xs) // len(xs)
    cy = sum(ys) // len(ys)
    w = (max(xs) - min(xs)) * 1.4
    h = (max(ys) - min(ys)) * 1.6

    x1 = int(max(cx - w // 2, 0))
    y1 = int(max(cy - h // 2, 0))
    x2 = int(min(cx + w // 2, img.width))
    y2 = int(min(cy + h // 2, img.height))

    box = (x1, y1, x2, y2)
    region = img.crop(box)

    # Яркость, контраст, glow
    region = ImageEnhance.Brightness(region).enhance(1.12)
    region = ImageEnhance.Contrast(region).enhance(1.18)
    glow = region.filter(ImageFilter.GaussianBlur(radius=2))
    region = Image.blend(region, glow, 0.25)

    # Плавная маска
    mask = Image.new("L", region.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, region.size[0], region.size[1]), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))

    base = img.crop(box)
    final = Image.composite(region, base, mask)
    img.paste(final, box)

    return img

def enhance_all_eyes(image: Image.Image, faces: list) -> Image.Image:
    """Улучшение глаз у всех обнаруженных лиц."""
    img = image.copy()
    for face in faces:
        if not hasattr(face, "landmark_2d_106"):
            continue
        landmarks = face.landmark_2d_106

        # Берём только реальные глазные области (левый и правый глаз)
        left_eye_points = [landmarks[i] for i in range(96, 102)]
        right_eye_points = [landmarks[i] for i in range(102, 108)]

        img = enhance_eye_by_center(img, left_eye_points)
        img = enhance_eye_by_center(img, right_eye_points)

    return img

def enhance_eye_by_center(image: Image.Image, points: list) -> Image.Image:
    """Улучшение глаза по центру landmarks с мягкой эллиптической маской."""
    img = image.copy()
    
    # Вычисляем центр и радиус
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    if not xs or not ys:
        return img

    cx = int(sum(xs) / len(xs))
    cy = int(sum(ys) / len(ys))
    rx = int((max(xs) - min(xs)) * 1.1)
    ry = int((max(ys) - min(ys)) * 1.6)

    x1 = max(cx - rx, 0)
    y1 = max(cy - ry, 0)
    x2 = min(cx + rx, image.width)
    y2 = min(cy + ry, image.height)
    box = (x1, y1, x2, y2)

    region = img.crop(box)

    # Улучшаем яркость и контраст
    region = ImageEnhance.Brightness(region).enhance(1.08)
    region = ImageEnhance.Contrast(region).enhance(1.15)

    # Добавляем мягкое свечение
    glow = region.filter(ImageFilter.GaussianBlur(radius=2.5))
    region = Image.blend(region, glow, 0.15)

    # Маска — мягкий эллипс
    mask = Image.new("L", region.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, region.size[0], region.size[1]), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=6))

    base = img.crop(box)
    final = Image.composite(region, base, mask)
    img.paste(final, box)

    return img

def enhance_all_eyes(image: Image.Image, faces: list) -> Image.Image:
    """Улучшение глаз у всех обнаруженных лиц."""
    img = image.copy()
    for face in faces:
        if not hasattr(face, "landmark_2d_106"):
            continue
        landmarks = face.landmark_2d_106

        # Берём только реальные глазные области (левый и правый глаз)
        left_eye_points = landmarks[96:102]   # 96–101
        right_eye_points = landmarks[102:106] # 102–105

        img = enhance_eye_by_center(img, left_eye_points)
        img = enhance_eye_by_center(img, right_eye_points)

    return img

    return img
def apply_skin_warmth_overlay(image: Image.Image, intensity: float = 0.035) -> Image.Image:
    """Добавление тёплого, мягкого тона кожи как в Remini."""
    img = image.copy()

    # Тёплый оттенок: можно варьировать от светло-персикового до розового
    warm_base = np.array([
        [255, 235, 210],  # мягкий персик
        [255, 220, 200],  # бежево-розовый
        [255, 240, 215],  # молочный
    ])
    # Рандомный выбор оттенка
    tone = tuple(warm_base[np.random.randint(0, len(warm_base))])

    overlay = Image.new("RGB", img.size, tone)
    img = Image.blend(img, overlay, intensity)

    # Дополнительное мягкое свечение
    glow = img.filter(ImageFilter.GaussianBlur(radius=3))
    img = Image.blend(img, glow, 0.05)

    return img

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

def apply_full_skin_glow(image: Image.Image, face_data) -> Image.Image:
    """Молочно-гладкий персиковый glow по всему телу как у глаза 😍"""
    img = image.copy()

    x1, y1, x2, y2 = map(int, face_data.bbox)
    width, height = img.size

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2 + int((y2 - y1) * 0.5)

    ellipse_width = int((x2 - x1) * 3.6)
    ellipse_height = int((y2 - y1) * 4.5)

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([
        max(center_x - ellipse_width // 2, 0),
        max(center_y - ellipse_height // 2, 0),
        min(center_x + ellipse_width // 2, width),
        min(center_y + ellipse_height // 2, height)
    ], fill=160)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=60))

    # Эффект — как у глаза:
    enhanced = img.copy()
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.12)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.15)

    glow = enhanced.filter(ImageFilter.GaussianBlur(radius=4))
    enhanced = Image.blend(enhanced, glow, 0.25)

    overlay = Image.new("RGB", img.size, (255, 240, 225))  # светлый персик
    enhanced = Image.blend(enhanced, overlay, 0.04)

    return Image.composite(enhanced, img, mask)

def apply_full_skin_glow_match_eye(image: Image.Image) -> Image.Image:
    """Применяет мягкий тёплый тон и свечение ко всему изображению."""
    img = image.copy()
    
    # Теплый светлый слой (тот самый «глазной»)
    warm_glow = Image.new("RGB", img.size, (255, 230, 200))
    glow_overlay = Image.blend(img, warm_glow, 0.07)

    # Повышение яркости и мягкости
    brightened = ImageEnhance.Brightness(glow_overlay).enhance(1.07)
    softened = brightened.filter(ImageFilter.GaussianBlur(radius=1.2))

    # Комбинируем с оригиналом для сохранения резкости
    final = Image.blend(img, softened, 0.35)

    return final

def apply_full_glow_to_all(image: Image.Image) -> Image.Image:
    """Нанесение полного персикового glow по всей фотографии — как внутри глаза."""
    img = image.copy()

    # Этап 1: Базовое усиление
    enhanced = ImageEnhance.Brightness(img).enhance(1.10)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.15)

    # Этап 2: Glow эффект
    glow = enhanced.filter(ImageFilter.GaussianBlur(radius=4))
    enhanced = Image.blend(enhanced, glow, 0.25)

    # Этап 3: Тёплый персиковый налёт
    overlay = Image.new("RGB", img.size, (255, 240, 225))
    final = Image.blend(enhanced, overlay, 0.04)

    return final

def apply_true_eye_glow_to_all(image: Image.Image) -> Image.Image:
    """Glow как у глаза — + восстановление чёткости."""
    img = image.copy()

    bright = ImageEnhance.Brightness(img).enhance(1.05)
    contrast = ImageEnhance.Contrast(bright).enhance(1.2)

    glow = contrast.filter(ImageFilter.GaussianBlur(radius=4))
    blended = Image.blend(contrast, glow, 0.25)

    overlay = Image.new("RGB", img.size, (255, 240, 225))
    final = Image.blend(blended, overlay, 0.04)

    # ⬅️ ВОТ ОН — шаг резкости
    final = final.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))

    return final

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
    
    if strength == 'strong':
        # Создаем несколько версий с разным уровнем размытия
        blur1 = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        blur2 = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        # Находим края
        edge_mask = img.filter(ImageFilter.FIND_EDGES)
        edge_mask = edge_mask.convert('L')
        edge_mask = edge_mask.point(lambda x: 255 if x > 20 else 0)
        edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        # Создаем маску деталей
        detail_mask = img.filter(ImageFilter.DETAIL)
        detail_mask = detail_mask.convert('L')
        detail_mask = detail_mask.point(lambda x: 255 if x > 15 else 0)
        
        # Комбинируем маски
        final_mask = ImageChops.lighter(edge_mask, detail_mask)
        final_mask = final_mask.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        # Смешиваем версии
        result = Image.composite(img, blur1, final_mask)
        result = Image.composite(result, blur2, edge_mask)
        
        # Восстанавливаем детали
        result = result.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
    else:
        # Стандартное шумоподавление
        blurred = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        edge_mask = img.filter(ImageFilter.FIND_EDGES)
        edge_mask = edge_mask.convert('L')
        edge_mask = edge_mask.point(lambda x: 255 if x > 25 else 0)
        edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        result = Image.composite(img, blurred, edge_mask)
        result = result.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))

    return result

def apply_natural_glow(image: Image.Image) -> Image.Image:
    """Добавление естественного свечения."""
    # Создаем слой свечения
    glow = image.copy()
    glow = ImageEnhance.Brightness(glow).enhance(1.2)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=10))
    
    # Смешиваем с оригиналом
    return Image.blend(image, glow, 0.3)

def enhance_skin_and_hair(region: Image.Image, strength: float = 1.0) -> Image.Image:
    """Улучшение кожи и волос с созданием красивого, живого эффекта."""
    enhanced = region.copy()
    
    # Создаем теплый свет
    warm_light = Image.new('RGB', region.size, (255, 245, 235))
    enhanced = Image.blend(enhanced, warm_light, 0.1 * strength)
    
    # Добавляем немного розового оттенка для живости
    pink_glow = Image.new('RGB', region.size, (255, 240, 240))
    enhanced = Image.blend(enhanced, pink_glow, 0.05 * strength)
    
    # Увеличиваем яркость и контраст
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.15 * strength)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.08 * strength)
    
    # Добавляем легкое свечение
    glow = enhanced.copy()
    glow = ImageEnhance.Brightness(glow).enhance(1.2)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=5))
    enhanced = Image.blend(enhanced, glow, 0.2 * strength)
    
    return enhanced

def normalize_skin_tone(face_region: Image.Image) -> Image.Image:
    """Нормализация цвета кожи к естественному теплому белому."""
    # Анализируем текущий цвет кожи
    img_np = np.array(face_region)
    skin_mask = (img_np[:,:,0] > 60) & (img_np[:,:,1] > 60) & (img_np[:,:,2] > 60)
    if not np.any(skin_mask):
        return face_region
    
    skin_pixels = img_np[skin_mask]
    avg_color = np.mean(skin_pixels, axis=0)
    r, g, b = avg_color
    
    # Определяем наличие сильного цветового оттенка от фар
    is_car_light = (r/b > 1.4 or g/b > 1.4) and (r + g)/(2*b) > 1.3
    
    if is_car_light:
        # Для света фар - просто предотвращаем усиление цвета
        enhanced = face_region.copy()
        r, g, b = enhanced.split()
        
        # Слегка уменьшаем насыщенность цвета
        enhanced = ImageEnhance.Color(enhanced).enhance(0.95)
        
        # Маска для плавного перехода
        mask = Image.new('L', face_region.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([0, 0, face_region.width, face_region.height], fill=180)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=face_region.width//8))
        
        return Image.composite(enhanced, face_region, mask)
    
    return face_region

def enhance_person_region(image: Image.Image, face_data, scene_type: str = "day") -> Image.Image:
    """Улучшение только области человека."""
    if not face_data:
        return image
        
    img = image.copy()
    
    # Анализируем все лица на фото
    faces = face_analyzer.get(np.array(img))
    face_regions = []
    
    # Определяем тип освещения для всей сцены
    is_club_lighting = False
    if scene_type == "night":
        # Проверяем наличие клубного освещения
        r, g, b = img.split()
        r_mean = np.mean(np.array(r))
        g_mean = np.mean(np.array(g))
        b_mean = np.mean(np.array(b))
        is_club_lighting = b_mean > (r_mean + g_mean) / 2 + 10
    
    # Собираем информацию о всех лицах
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        face_region = img.crop((x1, y1, x2, y2))
        face_regions.append((face_region, (x1, y1, x2, y2)))
    
    # Создаем улучшенную версию
    enhanced = img.copy()
    
    if scene_type == "day":
        # Дневная обработка в стиле примера
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.05)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.08)
        
        # Добавляем "дорогой" эффект
        overlay = Image.new('RGB', enhanced.size, (255, 253, 250))
        enhanced = Image.blend(enhanced, overlay, 0.05)
        
    elif is_club_lighting:
        # Клубное освещение
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.15)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.12)
        
        # Сохраняем цветовое освещение фона
        bg_mask = Image.new('L', enhanced.size, 128)
        for _, (x1, y1, x2, y2) in face_regions:
            draw = ImageDraw.Draw(bg_mask)
            # Маска для лица с отступами
            padding = ((x2 - x1) * 0.2)
            draw.ellipse([
                x1 - padding, y1 - padding,
                x2 + padding, y2 + padding
            ], fill=0)
        bg_mask = bg_mask.filter(ImageFilter.GaussianBlur(radius=20))
        
        # Применяем улучшение фона
        bg_enhanced = ImageEnhance.Brightness(img).enhance(1.2)
        enhanced = Image.composite(enhanced, bg_enhanced, bg_mask)
        
    else:
        # Ночная обработка без клубного света
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.12)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.1)
    
    # Обрабатываем каждое лицо отдельно
    for face_region, (x1, y1, x2, y2) in face_regions:
        # Определяем область вокруг лица для плавного перехода
        padding_x = int((x2 - x1) * 0.3)
        padding_y = int((y2 - y1) * 0.3)
        
        face_area = enhanced.crop((
            max(0, x1 - padding_x),
            max(0, y1 - padding_y),
            min(enhanced.width, x2 + padding_x),
            min(enhanced.height, y2 + padding_y)
        ))
        
        # Улучшаем область лица
        if scene_type == "day":
            # Дневная обработка лица
            face_area = ImageEnhance.Brightness(face_area).enhance(1.03)
            face_area = ImageEnhance.Contrast(face_area).enhance(1.06)
            
            # Добавляем легкое сияние
            glow = face_area.filter(ImageFilter.GaussianBlur(radius=10))
            face_area = Image.blend(face_area, glow, 0.3)
            
        elif is_club_lighting:
            # Клубное освещение - более интенсивная обработка
            face_area = ImageEnhance.Brightness(face_area).enhance(1.07)
            face_area = ImageEnhance.Contrast(face_area).enhance(1.50)
            
            # Сохраняем детали
            face_area = face_area.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            
        else:
            # Ночная обработка
            face_area = ImageEnhance.Brightness(face_area).enhance(1.12)
            face_area = ImageEnhance.Contrast(face_area).enhance(1.12)
        
        # Создаем маску для плавного перехода
        mask = Image.new('L', face_area.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([
            padding_x - padding_x//2,
            padding_y - padding_y//2,
            padding_x + (x2 - x1) + padding_x//2,
            padding_y + (y2 - y1) + padding_y//2
        ], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=padding_x//3))
        
        # Вставляем обработанную область обратно
        enhanced_region = Image.composite(face_area, enhanced.crop((
            max(0, x1 - padding_x),
            max(0, y1 - padding_y),
            min(enhanced.width, x2 + padding_x),
            min(enhanced.height, y2 + padding_y)
        )), mask)
        
        enhanced.paste(enhanced_region, (
            max(0, x1 - padding_x),
            max(0, y1 - padding_y)
        ))
    
    # Финальные штрихи
    if scene_type == "day":
        # Добавляем "дорогое" качество
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    elif is_club_lighting:
        # Сохраняем атмосферу клуба
        enhanced = ImageEnhance.Color(enhanced).enhance(1.1)
    else:
        # Ночная обработка - баланс между качеством и естественностью
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=130, threshold=3))
    
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

def apply_daylight_enhancement(image: Image.Image, face_data) -> Image.Image:
    """Улучшение фотографии при дневном освещении."""
    return enhance_person_region(image, face_data, "day")

def apply_evening_enhancement(image: Image.Image, face_data) -> Image.Image:
    """Улучшение фотографии при вечернем/искусственном освещении."""
    return enhance_person_region(image, face_data, "evening")

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
    
    # Проверяем наличие синего/фиолетового освещения (клубное)
    is_club = b_mean > (r_mean + g_mean) / 2 + 15
    
    # Проверяем равномерность освещения
    std_dev = np.std(img_np)
    is_uniform = std_dev < 50
    
    if is_club or (brightness < 90 and has_colored_light):
        return "club"
    elif brightness > 160 and is_uniform:
        return "daylight"
    else:
        return "evening"

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
                "prompt_strength": 0.06,    # Умеренная сила промпта
                "num_inference_steps": 22,  # Оптимальное количество шагов
                "output_format": "png",
                "output_quality": 90,
                "go_fast": True,
                "lora_scale": 0.5,         # Средний уровень LoRA
                "extra_lora_scale": 0.07    # Умеренное дополнительное влияние
            }
        )
        
        # ✅ Сначала скачиваем и открываем результат
        response = requests.get(str(idnbeauty_result[0]))
        image_idn = Image.open(io.BytesIO(response.content)).convert("RGB")

    # ✅ Только теперь можно использовать image_idn
        img_np = np.array(image_idn)
        faces = face_analyzer.get(img_np)
        face = faces[0] if faces else None
        scene_type = classify_scene(image_idn)
        skin_tone = analyze_skin_tone(image_idn, face)

# 👁 Улучшаем глаза всем (если вернёшь обратно)
        # image_idn = enhance_all_eyes(image_idn, faces)

# 🌡 Потепление тона кожи
        image_idn = apply_skin_warmth_overlay(image_idn, intensity=0.035)

# 💾 Сохраняем лицо ДО свечения, если оно есть
        if face:
            x1, y1, x2, y2 = map(int, face.bbox)
            face_region = image_idn.crop((x1, y1, x2, y2))

    # ✨ Корректируем только лицо (до glow)
            face_region = normalize_skin_tone(face_region)

# ✨ Glow на всё тело (до вставки лица обратно)
        image_idn = apply_full_glow_to_all(image_idn)
        image_idn = apply_true_eye_glow_to_all(image_idn)

# 🧩 Возвращаем лицо обратно, чтобы оно не стало мыльным
        if face:
            image_idn.paste(face_region, (x1, y1))

# 🧠 Финальное улучшение по сцене
        final_image = enhance_person_region(image_idn, face, "day" if scene_type != "night" else "evening")

# 🌈 Финальный glow (тот самый как в круге)
        final_image = apply_full_skin_glow_match_eye(final_image)
  
# 💾 Сохраняем
        final_bytes = io.BytesIO()
        final_image.save(final_bytes, format="JPEG", quality=100, subsampling=0)
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