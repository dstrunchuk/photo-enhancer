import replicate
import requests
import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import io
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime
import numpy as np

# Инициализация клиента Replicate
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Инициализация распознавания лица
face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0)

def has_face(image_path: str) -> bool:
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    faces = face_analyzer.get(img_np)
    return len(faces) > 0

def conditional_brightness(image: Image.Image) -> Image.Image:
    # Переводим в numpy и считаем среднюю яркость (от 0 до 255)
    grayscale = image.convert("L")
    avg_brightness = np.mean(np.array(grayscale))

    # Логика: если темное — усиливаем сильнее, если светлое — мягче
    if avg_brightness < 100:
        factor = 1.50  # тёмное фото — сильно осветляем
    elif avg_brightness < 160:
        factor = 1.35  # средняя яркость — умеренно
    else:
        factor = 1.20  # уже светлое — слегка

    return ImageEnhance.Brightness(image).enhance(factor)

# Лёгкая цветокоррекция + акцент на чёткость
def apply_final_polish(image: Image.Image) -> Image.Image:
    image = conditional_brightness(image)
    image = ImageEnhance.Contrast(image).enhance(1.10)
    image = ImageEnhance.Color(image).enhance(1.10)
    image = ImageEnhance.Sharpness(image).enhance(1.50)  # Сильнее подчёркиваем резкость
    return image

# Основная функция
async def enhance_image(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image.save("input.jpg")

    if not has_face("input.jpg"):
        raise Exception("Лицо не обнаружено. Пожалуйста, загрузите чёткий портрет.")

    # Шаг 1 — IDNBeauty (мягкая ретушь без изменений черт лица)
    try:
        idnbeauty_result = replicate.run(
            "torrikabe-ai/idnbeauty:5f994656b3b88df2e21a3cf0a81371d66bd6ff45171f3e5618bb314bdc8b64b1",
            input={
                "image": open("input.jpg", "rb"),
                "prompt": (
                    "Subtle and natural retouching. Lightly reduce under-eye bags and strong shadows. "
                    "Keep skin texture, identity, and facial features unchanged. No artificial edits or smoothing."
                ),
                "model": "dev",
                "guidance_scale": 0.5,
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
        print(f"IDNBeauty failed: {e}")
        raise Exception("Ошибка при обработке IDNBeauty")

    # Шаг 2 — Цветокоррекция и подчёркивание деталей
    final_image = apply_final_polish(image_idn)

    final_bytes = io.BytesIO()
    final_image.save(final_bytes, format="JPEG", quality=99, subsampling=0)
    final_bytes.seek(0)
    return final_bytes.read()