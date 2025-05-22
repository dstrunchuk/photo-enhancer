import replicate
import requests
import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import io
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime

# Инициализация клиента Replicate
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Инициализация распознавания лица
face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0)

# Проверка наличия лица
def has_face(image_path: str) -> bool:
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    faces = face_analyzer.get(img_np)
    return len(faces) > 0

# Эффект студийного света и цветокоррекция
def apply_studio_polish(image: Image.Image) -> Image.Image:
    shadow_layer = image.filter(ImageFilter.GaussianBlur(radius=20))
    shadow_layer = ImageEnhance.Brightness(shadow_layer).enhance(1.3)
    image = Image.blend(image, shadow_layer, alpha=0.25)

    image = ImageEnhance.Brightness(image).enhance(1.10)
    image = ImageEnhance.Contrast(image).enhance(1.08)
    image = ImageEnhance.Color(image).enhance(1.04)
    image = ImageEnhance.Sharpness(image).enhance(1.2)
    return image

# Основная функция
async def enhance_image(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image.save("input.jpg")

    if not has_face("input.jpg"):
        raise Exception("Лицо не обнаружено. Пожалуйста, загрузите чёткий портрет.")

    # Шаг 1 — IDNBeauty вместо CodeFormer
    try:
        idnbeauty_result = replicate.run(
            "torrikabe-ai/idnbeauty:5f994656b3b88df2e21a3cf0a81371d66bd6ff45171f3e5618bb314bdc8b64b1",
            input={
                "image": open("input.jpg", "rb"),
                "prompt": (
                    "Subtle and natural skin retouching. Lightly reduce under-eye bags and strong shadows. "
                    "Preserve full facial texture, structure, identity, and natural expression. Do not apply makeup."
                ),
                "model": "dev",
                "guidance_scale": 0.5,
                "prompt_strength": 0.05,
                "num_inference_steps": 28,
                "output_format": "png",
                "output_quality": 85,
                "go_fast": False,
                "lora_scale": 0.85,
                "extra_lora_scale": 0.15
            }
        )
        response = requests.get(str(idnbeauty_result[0]))
        image_cf = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"IDNBeauty failed: {e}")
        raise Exception("Ошибка при обработке IDNBeauty")

    # Шаг 2 — Финальная обработка
    final_image = apply_studio_polish(image_cf)

    final_bytes = io.BytesIO()
    final_image.save(final_bytes, format="JPEG", quality=99, subsampling=0)
    final_bytes.seek(0)
    return final_bytes.read()