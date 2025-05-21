import replicate
import requests
import os
from PIL import Image
import io
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime

# Инициализация Replicate
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Инициализация распознавания лиц
face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0)

# Проверка наличия лица
def has_face(image_path: str) -> bool:
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    faces = face_analyzer.get(img_np)
    return len(faces) > 0

# Сжатие и ресайз
def compress_and_resize(image_path: str, output_path: str, max_size=1600):
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)
    image.save(output_path, format="JPEG", quality=90, optimize=True)

# Основная функция
async def enhance_image(image_bytes: bytes) -> bytes:
    # Сохраняем оригинал
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.save("input.jpg")

    if not has_face("input.jpg"):
        raise Exception("Лицо не обнаружено. Пожалуйста, загрузите чёткий портрет.")

    # Шаг 1 — GFPGAN
    gfpgan_url = replicate.run(
        "tencentarc/gfpgan",
        input={"img": open("input.jpg", "rb")}
    )
    gfpgan_img = requests.get(gfpgan_url[0])
    with open("gfpgan_output.jpg", "wb") as f:
        f.write(gfpgan_img.content)

    # Сжатие перед CodeFormer
    compress_and_resize("gfpgan_output.jpg", "gfpgan_resized.jpg")

    # Шаг 2 — CodeFormer
    try:
        codeformer_url = replicate.run(
            "sczhou/codeformer",
            input={
                "image": open("gfpgan_resized.jpg", "rb"),
                "upscale": 2,
                "face_upsample": True,
                "background_enhance": True,
                "codeformer_fidelity": 0.8
            }
        )
        codeformer_img = requests.get(codeformer_url[0])
        with open("codeformer_output.jpg", "wb") as f:
            f.write(codeformer_img.content)
    except Exception as e:
        print(f"CodeFormer failed: {e} — returning GFPGAN result.")
        return gfpgan_img.content

    # Шаг 3 — Real-ESRGAN
    try:
        realesrgan_url = replicate.run(
            "nightmareai/real-esrgan",
            input={
                "image": open("codeformer_output.jpg", "rb"),
                "scale": 1,  # не увеличиваем размер
                "face_enhance": False
            }
        )
        realesrgan_img = requests.get(realesrgan_url[0])
        final_image = Image.open(io.BytesIO(realesrgan_img.content)).convert("RGB")
    except Exception as e:
        print(f"Real-ESRGAN failed: {e} — returning CodeFormer result.")
        final_image = Image.open("codeformer_output.jpg").convert("RGB")

    # Возвращаем финальное изображение
    img_bytes = io.BytesIO()
    final_image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes.read()