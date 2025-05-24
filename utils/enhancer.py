import replicate
import requests
import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import io
import numpy as np
import uuid
from insightface.app import FaceAnalysis
import onnxruntime
from PIL import ImageDraw

# Инициализация клиента Replicate
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Инициализация распознавания лица
face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0)

def enhance_eyes_and_lips(image: Image.Image, face_data) -> Image.Image:
    if not face_data or not hasattr(face_data, "landmark_2d_106"):
        return image
    img = image.copy()
    landmarks = face_data.landmark_2d_106

    def get_mask_bbox(points, padding=4):
        x = [int(p[0]) for p in points]
        y = [int(p[1]) for p in points]
        x1, x2 = max(min(x) - padding, 0), min(max(x) + padding, img.width)
        y1, y2 = max(min(y) - padding, 0), min(max(y) + padding, img.height)
        return (x1, y1, x2, y2)

    for region_points, enhancer in [
        (landmarks[33:42], lambda i: ImageEnhance.Brightness(ImageEnhance.Sharpness(i).enhance(2)).enhance(1.2)),
        (landmarks[42:51], lambda i: ImageEnhance.Brightness(ImageEnhance.Sharpness(i).enhance(2)).enhance(1.2)),
        (landmarks[70:89], lambda i: ImageEnhance.Color(i).enhance(1.35))
    ]:
        bbox = get_mask_bbox(region_points)
        sub = img.crop(bbox)
        sub_enhanced = enhancer(sub)
        mask = Image.new("L", sub.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, sub.size[0], sub.size[1]), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=4))
        img.paste(sub_enhanced, bbox[:2], mask)
    return img

# Проверка наличия лица
def has_face(image_path: str) -> bool:
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    faces = face_analyzer.get(img_np)
    return len(faces) > 0

def apply_premium_smooth(image: Image.Image) -> Image.Image:
    base = image.copy()
    blur = base.filter(ImageFilter.GaussianBlur(radius=1.5))
    glow = ImageEnhance.Brightness(blur).enhance(1.03)
    mix = Image.blend(base, glow, 0.20)
    return mix

# Проверка яркости именно на текущем изображении
def get_face_brightness_live(image: Image.Image) -> float:
    img_np = np.array(image)
    faces = face_analyzer.get(img_np)
    if not faces:
        return np.mean(np.array(image.convert("L")))
    face = faces[0]
    x1, y1, x2, y2 = map(int, face.bbox)
    face_crop = image.crop((x1, y1, x2, y2)).convert("L")
    return np.mean(np.array(face_crop))

def is_dark_photo(image: Image.Image) -> bool:
    gray = image.convert("L")
    avg_brightness = np.array(gray).mean()
    return avg_brightness < 50

def apply_subject_lighting(image: Image.Image) -> Image.Image:
    # Мягкое "осветление тела" — эмуляция студийного света
    body_layer = image.filter(ImageFilter.GaussianBlur(radius=30))
    body_layer = ImageEnhance.Brightness(body_layer).enhance(1.25)
    result = Image.blend(image, body_layer, alpha=0.25)

    # Уменьшение красного оттенка (часто бывает на ночных фото)
    r, g, b = result.split()
    r = r.point(lambda i: i * 0.95)
    result = Image.merge("RGB", (r, g, b))
    return result

# Регулируем осветление по яркости лица
def conditional_brightness(image: Image.Image) -> Image.Image:
    avg_brightness = get_face_brightness_live(image)
    if avg_brightness > 145:
        brightness_factor = 1.00
    elif avg_brightness > 120:
        brightness_factor = 1.08
    else:
        brightness_factor = np.clip(1.4 - (avg_brightness - 80) * 0.00375, 1.1, 1.4)
    return ImageEnhance.Brightness(image).enhance(brightness_factor)

def add_face_glow(image: Image.Image, face_data) -> Image.Image:
    if not face_data:
        return image
    x1, y1, x2, y2 = map(int, face_data.bbox)
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([x1, y1, x2, y2], fill=255)
    glow_mask = mask.filter(ImageFilter.GaussianBlur(radius=40))
    bright = ImageEnhance.Brightness(image).enhance(1.18)
    return Image.composite(bright, image, glow_mask)

def analyze_skin_tone(image: Image.Image, face_data) -> str:
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
    
def apply_soft_filter(image: Image.Image) -> Image.Image:
    base = image.copy()

    # Мягкое размытие
    blurred = base.filter(ImageFilter.GaussianBlur(radius=2.2))

    # Усиление яркости и света
    brightened = ImageEnhance.Brightness(blurred).enhance(1.07)
    contrast = ImageEnhance.Contrast(brightened).enhance(1.03)

    # Смешиваем с оригиналом — даёт эффект glow без потери деталей
    result = Image.blend(base, contrast, alpha=0.25)

    return result
    
def adjust_by_skin_tone(image: Image.Image, tone: str) -> Image.Image:
    img = image.copy()
    if tone == "pale":
        img = ImageEnhance.Color(img).enhance(1.15)
        overlay = Image.new("RGB", img.size, (35, 25, 20))
        img = Image.blend(img, overlay, 0.06)
    elif tone == "red":
        r, g, b = img.split()
        r = r.point(lambda i: i * 0.93)
        img = Image.merge("RGB", (r, g, b))
    elif tone == "yellow":
        r, g, b = img.split()
        g = g.point(lambda i: i * 0.90)
        img = Image.merge("RGB", (r, g, b))
    elif tone == "cold":
        overlay = Image.new("RGB", img.size, (50, 30, 20))
        img = Image.blend(img, overlay, 0.10)
    return img



# Цветокоррекция с адаптивной яркостью
def apply_final_polish(image: Image.Image) -> Image.Image:
    image = conditional_brightness(image)
    image = ImageEnhance.Contrast(image).enhance(1.10)
    image = ImageEnhance.Color(image).enhance(1.10)
    image = ImageEnhance.Sharpness(image).enhance(1.40)
    return image

# Классификация сцены по фото
def classify_scene(image: Image.Image) -> str:
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
    
def apply_background_blur(image: Image.Image, face_data) -> Image.Image:
    if not face_data:
        return image

    img = image.copy()

    # Создание маски лица
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = map(int, face_data.bbox)
    draw.rectangle([x1, y1, x2, y2], fill=255)

    # Размытие всей картинки
    blurred = img.filter(ImageFilter.GaussianBlur(radius=3))

    # Вставляем обратно лицо из оригинала
    img.paste(image, mask=mask)

    # Опционально: мягкое свечение вокруг лица
    glow_mask = mask.filter(ImageFilter.GaussianBlur(radius=12))
    img = Image.composite(image, blurred, glow_mask)

    return img

# Применить эффекты по сценарию и маске лица
def apply_scenario(image: Image.Image, face_data, scene_type: str) -> Image.Image:
    img = image.copy()

    # Создаём маску лица
    face_mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(face_mask)
    if face_data:
        x1, y1, x2, y2 = map(int, face_data.bbox)
        draw.rectangle([x1, y1, x2, y2], fill=255)

    # Обработка по сценарию
    if scene_type == "night":
        img = apply_subject_lighting(img)  # уже делает лёгкий glow и осветление
    elif scene_type == "cold_white_light":
        # Лёгкий тёплый фильтр
        warm_overlay = Image.new("RGB", img.size, (30, 20, 10))
        img = Image.blend(img, warm_overlay, alpha=0.08)

        # Уменьшение резкости в лице
        blurred = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        img.paste(blurred, mask=face_mask)

    elif scene_type == "overexposed":
        img = ImageEnhance.Contrast(img).enhance(0.95)
        img = ImageEnhance.Brightness(img).enhance(0.95)

    # Дополнительно: осветлить лицо, если оно темное
    if face_data:
        x1, y1, x2, y2 = map(int, face_data.bbox)
        face_region = img.crop((x1, y1, x2, y2))
        bright_face = ImageEnhance.Brightness(face_region).enhance(1.12)
        img.paste(bright_face, (x1, y1))

    return img

def apply_soft_filter(image: Image.Image, intensity: str = "normal") -> Image.Image:
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

# Основная функция
async def enhance_image(image_bytes: bytes, user_prompt: str = "") -> bytes:
    # Открытие изображения
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)

    temp_filename = f"{uuid.uuid4()}.jpg"
    image.save(temp_filename)

    # Проверка лица до IDNBeauty
    if not has_face(temp_filename):
        os.remove(temp_filename)
        raise Exception("Лицо не обнаружено. Пожалуйста, загрузите чёткий портрет.")

    # Запуск IDNBeauty
    try:
        prompt = (
            "Subtle and natural retouching. Lightly reduce under-eye bags and strong shadows. "
            "Keep skin texture, identity, and facial features unchanged. Do not alter eyes, eyelashes, or lips. "
            "No artificial edits, no smoothing, no additions."
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
    except Exception as e:
        os.remove(temp_filename)
        raise Exception(f"Ошибка при обработке IDNBeauty: {e}")

    os.remove(temp_filename)

    # Анализ лица
    img_np = np.array(image_idn)
    faces = face_analyzer.get(img_np)
    face = faces[0] if faces else None

    # Классификация сцены и тона
    scene_type = classify_scene(image_idn)
    skin_tone = analyze_skin_tone(image_idn, face)

    # Применение сценариев
    image_idn = apply_scenario(image_idn, face, scene_type)
    image_idn = apply_background_blur(image_idn, face)
    image_idn = enhance_eyes_and_lips(image_idn, face)
    image_idn = adjust_by_skin_tone(image_idn, skin_tone)
    image_idn = add_face_glow(image_idn, face)

    # Ночной фильтр освещения
    if is_dark_photo(image_idn):
        image_idn = apply_subject_lighting(image_idn)

    # Финальная цветокоррекция и мягкость
    final_image = apply_final_polish(image_idn)
    soft_intensity = "strong" if scene_type == "night" else "normal"
    final_image = apply_soft_filter(final_image, intensity=soft_intensity)

    # Сохранение
    final_bytes = io.BytesIO()
    final_image = apply_premium_smooth(final_image)
    final_image.save(final_bytes, format="JPEG", quality=99, subsampling=0)
    final_bytes.seek(0)
    return final_bytes.read()