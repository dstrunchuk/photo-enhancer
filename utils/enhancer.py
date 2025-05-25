# =============================================================================
# Импорты и конфигурация
# =============================================================================
import replicate
import requests
import zipfile
import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
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
    """Мягкое осветление кожи и волос."""
    img = image.copy()

    x1, y1, x2, y2 = map(int, face_data.bbox)
    width, height = img.size

    margin_x = int((x2 - x1) * 0.8)
    margin_y = int((y2 - y1) * 1.2)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    ex1 = max(center_x - margin_x, 0)
    ey1 = max(center_y - margin_y, 0)
    ex2 = min(center_x + margin_x, width)
    ey2 = min(center_y + margin_y, height)

    region = img.crop((ex1, ey1, ex2, ey2))

    glow = region.filter(ImageFilter.GaussianBlur(radius=10))
    glow = ImageEnhance.Brightness(glow).enhance(1.12)
    overlay = Image.new("RGB", region.size, (250, 235, 220))
    blended = Image.blend(glow, overlay, 0.04)

    mask = Image.new("L", region.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, region.size[0], region.size[1]), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=15))

    base = img.crop((ex1, ey1, ex2, ey2))
    final_region = Image.composite(blended, base, mask)
    img.paste(final_region, (ex1, ey1))

    return img

def enhance_face_lighting(image: Image.Image, face_data) -> Image.Image:
    """Улучшение освещения лица."""
    if not face_data:
        return image

    img = image.copy()
    x1, y1, x2, y2 = map(int, face_data.bbox)

    face_crop = img.crop((x1, y1, x2, y2))
    bright_face = ImageEnhance.Brightness(face_crop).enhance(1.12)
    warm_overlay = Image.new("RGB", bright_face.size, (255, 230, 200))
    enhanced = Image.blend(bright_face, warm_overlay, 0.06)
    img.paste(enhanced, (x1, y1))
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
            "Soft natural face enhancement. "
            "Do not touch eyes, pupils, eyelashes, eyelids, eyeliner, eye direction, or makeup. "
            "Preserve original lashes, gaze, and eye shape. "
            "Do not sharpen. Do not smooth. Do not change expression. "
            "Only slightly brighten and clean the skin. "
        )
        if user_prompt:
            prompt = user_prompt

        idnbeauty_result = replicate.run(
            "torrikabe-ai/idnbeauty:5f994656b3b88df2e21a3cf0a81371d66bd6ff45171f3e5618bb314bdc8b64b1",
            input={
                "image": open(temp_filename, "rb"),
                "prompt": prompt,
                "model": "dev",
                "guidance_scale": 0.6,
                "prompt_strength": 0.10,
                "num_inference_steps": 24,
                "output_format": "png",
                "output_quality": 90,
                "go_fast": True,
                "lora_scale": 0.8,
                "extra_lora_scale": 0.12
            }
        )

        response = requests.get(str(idnbeauty_result[0]))
        image_idn = Image.open(io.BytesIO(response.content)).convert("RGB")

        img_np = np.array(image_idn)
        faces = face_analyzer.get(img_np)
        face = faces[0] if faces else None
        scene_type = classify_scene(image_idn)
        skin_tone = analyze_skin_tone(image_idn, face)

        image_idn = adjust_by_skin_tone(image_idn, skin_tone)
        final_image = apply_final_polish(image_idn)

        if face:
            final_image = lighten_skin_and_hair_only(final_image, face)

        if scene_type == "night" and face:
            final_image = enhance_face_lighting(final_image, face)

        final_bytes = io.BytesIO()
        final_image.save(final_bytes, format="JPEG", quality=99, subsampling=0)
        final_bytes.seek(0)

        return final_bytes.read()

    except Exception as e:
        os.remove(temp_filename)
        raise Exception(f"Ошибка обработки IDNBeauty или постобработки: {e}")

async def enhance_image_remini(image_bytes: bytes) -> bytes:
    """Улучшение изображения в стиле Remini."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)

    temp_filename = f"{uuid.uuid4()}.jpg"
    image.save(temp_filename)

    if not has_face(temp_filename):
        os.remove(temp_filename)
        raise Exception("Лицо не обнаружено. Пожалуйста, загрузите чёткий портрет.")

    try:
        enhanced_result = replicate.run(
            "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b",
            input={
                "image": open(temp_filename, "rb"),
                "scale": 2,
                "face_enhance": True,
            }
        )

        response = requests.get(enhanced_result)
        enhanced_image = Image.open(io.BytesIO(response.content)).convert("RGB")

        img_np = np.array(enhanced_image)
        faces = face_analyzer.get(img_np)
        face = faces[0] if faces else None

        if face:
            enhanced_image = enhance_face_lighting(enhanced_image, face)
            enhanced_image = apply_soft_filter(enhanced_image, "normal")
            skin_tone = analyze_skin_tone(enhanced_image, face)
            enhanced_image = adjust_by_skin_tone(enhanced_image, skin_tone)

        final_image = apply_final_polish(enhanced_image)

        final_bytes = io.BytesIO()
        final_image.save(final_bytes, format="JPEG", quality=95, subsampling=0)
        final_bytes.seek(0)

        os.remove(temp_filename)
        return final_bytes.read()

    except Exception as e:
        os.remove(temp_filename)
        raise Exception(f"Ошибка при обработке фотографии: {e}")