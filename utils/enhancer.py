import replicate
import requests
import os
from PIL import Image, ImageOps, ImageEnhance
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

# Сжатие и уменьшение изображения
def compress_and_resize(image_path: str, output_path: str, max_size=1600):
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)
    image.save(output_path, format="JPEG", quality=90, optimize=True)

# Цветокоррекция
def apply_color_correction(image: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.20)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.05)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.06)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.03)
    return image

# Основная функция
async def enhance_image(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image.save("input.jpg")

    if not has_face("input.jpg"):
        raise Exception("Лицо не обнаружено. Пожалуйста, загрузите чёткий портрет.")

    # Шаг 1 — GFPGAN (новая версия с безопасным масштабом и сжатием)
    try:
        gfpgan_url = replicate.run(
            "xinntao/gfpgan:6129309904ce4debfde78de5c209bce0022af40e197e132f08be8ccce3050393",
            input={
                "img": open("input.jpg", "rb"),
                "scale": 1,  # уменьшено до 1, чтобы не перегружать
               "version": "v1.4"
            }
        )
        gfpgan_img = requests.get(gfpgan_url)
    
    # Безопасное открытие и сжатие результата
        gfpgan_image = Image.open(io.BytesIO(gfpgan_img.content)).convert("RGB")

    # Ограничим изображение по общей площади (например, 2500000 пикселей)
        if gfpgan_image.width * gfpgan_image.height > 2500000:
            scale = (2500000 / (gfpgan_image.width * gfpgan_image.height)) ** 0.5
            new_size = (int(gfpgan_image.width * scale), int(gfpgan_image.height * scale))
            gfpgan_image = gfpgan_image.resize(new_size, Image.LANCZOS)

    # Сохраняем с оптимизированным JPEG
        gfpgan_image.save("gfpgan_output.jpg", format="JPEG", quality=90, optimize=True)

    # Далее стандартное уменьшение (если нужно для CodeFormer)
        compress_and_resize("gfpgan_output.jpg", "gfpgan_resized.jpg")

    except Exception as e:
        print(f"GFPGAN failed: {e}")
        raise Exception("Ошибка при обработке GFPGAN")

    # Шаг 2 — CodeFormer (обновлённая модель)
    try:
        codeformer_url = replicate.run(
            "lucataco/codeformer:78f2bab438ab0ffc85a68cdfd316a2ecd3994b5dd26aa6b3d203357b45e5eb1b",
            input={
                "image": open("gfpgan_resized.jpg", "rb"),
                "upscale": 1,
                "face_upsample": True,
                "background_enhance": True,
                "codeformer_fidelity": 0.8
            }
        )
        codeformer_img = requests.get(codeformer_url)
        image_cf = Image.open(io.BytesIO(codeformer_img.content)).convert("RGB")
        image_cf.save("codeformer_output.jpg", format="JPEG", quality=95)
    except Exception as e:
        print(f"CodeFormer failed: {e} — returning GFPGAN result.")
        return gfpgan_img.content
    
        # Шаг 3 — Ретушь кожи
    try:
        skin_retouch_url = replicate.run(
            "torrikabe-ai/idnbeauty:latest",
            input={
                "image": open("codeformer_output.jpg", "rb"),
                "prompt": (
                    "Softly smooth skin and remove facial creases, blemishes, and harsh shadows while fully preserving the face shape, identity, and expression. "
                    "No makeup, no exaggeration. Keep the photo realistic and subtle."
                ),
                "model": "dev",
                "guidance_scale": 1,
                "prompt_strength": 0.1,
                "num_inference_steps": 28,
                "output_format": "png",
                "output_quality": 80,
                "go_fast": False,
                "lora_scale": 0.94,
                "extra_lora_scale": 0.22
            }
        )
        skin_retouch_img = requests.get(str(skin_retouch_url[0]))
        final_image = Image.open(io.BytesIO(skin_retouch_img.content)).convert("RGB")
    except Exception as e:
        print(f"Skin retouching failed: {e} — returning CodeFormer result.")
        final_image = Image.open("codeformer_output.jpg").convert("RGB")

    # Шаг 4 — Color Correction (УЖЕ ПОСЛЕ ретуши)
    final_image = apply_color_correction(final_image)

    # Финальный результат
    final_bytes = io.BytesIO()
    final_image.save(final_bytes, format="JPEG")
    final_bytes.seek(0)
    return final_bytes.read()