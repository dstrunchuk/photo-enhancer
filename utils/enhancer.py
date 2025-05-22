import replicate
import requests
import os
from PIL import Image, ImageOps, ImageEnhance
import io
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime
import matplotlib.colors

def lighten_shadows(image: Image.Image, factor=1.2) -> Image.Image:
    img_np = np.array(image).astype(np.float32) / 255.0
    hsv = matplotlib.colors.rgb_to_hsv(img_np)
    v = hsv[..., 2]
    mask = v < 0.5
    v[mask] = np.clip(v[mask] * factor, 0, 1.0)
    hsv[..., 2] = v
    img_np = matplotlib.colors.hsv_to_rgb(hsv)
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)

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

# Цветокоррекция с теплом и улучшенной резкостью
def apply_color_correction(image: Image.Image) -> Image.Image:
    image = lighten_shadows(image, factor=1.35)  # Осветляем тени
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.15)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.04)
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

    # Шаг 1 — CodeFormer (обновлённая модель)
    try:
        codeformer_url = replicate.run(
            "lucataco/codeformer:78f2bab438ab0ffc85a68cdfd316a2ecd3994b5dd26aa6b3d203357b45e5eb1b",
            input={
                "image": open("input.jpg", "rb"),
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
        print(f"CodeFormer failed: {e}")
        raise Exception("Ошибка при обработке CodeFormer")

    # Шаг 2 — Ретушь кожи
    try:
        skin_retouch_url = replicate.run(
            "torrikabe-ai/idnbeauty:5f994656b3b88df2e21a3cf0a81371d66bd6ff45171f3e5618bb314bdc8b64b1",
            input={
                "image": open("codeformer_output.jpg", "rb"),
                "prompt": (
                    "Apply soft and minimal skin retouching. Remove only visible facial shadows and deep creases. "
                    "Do not change eyes, lips, or face shape. Preserve all original facial features and textures. "
                    "The result must look natural and realistic with no visible edits."
                ),
                "model": "dev",
                "guidance_scale": 0.6,
                "prompt_strength": 0.05,
                "num_inference_steps": 28,
                "output_format": "png",
                "output_quality": 80,
                "go_fast": False,
                "lora_scale": 0.80,
                "extra_lora_scale": 0.15
            }
        )
        skin_retouch_img = requests.get(str(skin_retouch_url[0]))
        final_image = Image.open(io.BytesIO(skin_retouch_img.content)).convert("RGB")
    except Exception as e:
        print(f"Skin retouching failed: {e} — returning CodeFormer result.")
        final_image = Image.open("codeformer_output.jpg").convert("RGB")

    # Шаг 3 — Цветокоррекция
    final_image = apply_color_correction(final_image)

    # Возврат результата
    final_bytes = io.BytesIO()
    final_image.save(final_bytes, format="JPEG", quality=99, subsampling=0)
    final_bytes.seek(0)
    return final_bytes.read()