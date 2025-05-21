import replicate
import requests
import os
from PIL import Image, ImageOps
import io
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime

# Replicate API
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

# Уменьшение и сжатие изображения (для CodeFormer)
def compress_and_resize(image_path: str, output_path: str, max_size=1600):
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)
    image.save(output_path, format="JPEG", quality=90, optimize=True)

# Основная функция
async def enhance_image(image_bytes: bytes) -> bytes:
    # Сохраняем оригинал и извлекаем ориентацию
    image = Image.open(io.BytesIO(image_bytes))
    exif = image.getexif()
    orientation = exif.get(274)  # EXIF orientation tag
    image.save("input.jpg")

    if not has_face("input.jpg"):
        raise Exception("Лицо не обнаружено. Пожалуйста, загрузите чёткий портрет.")

    # Шаг 1 — GFPGAN
    gfpgan_url = replicate.run(
        "tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
        input={"img": open("input.jpg", "rb")}
    )
    gfpgan_img = requests.get(gfpgan_url)
    with open("gfpgan_output.jpg", "wb") as f:
        f.write(gfpgan_img.content)

    compress_and_resize("gfpgan_output.jpg", "gfpgan_resized.jpg")

    # Шаг 2 — CodeFormer
    try:
        codeformer_url = replicate.run(
            "sczhou/codeformer:cc4956dd26fa5a7185d5660cc9100fab1b8070a1d1654a8bb5eb6d443b020bb2",
            input={
                "image": open("gfpgan_resized.jpg", "rb"),
                "upscale": 2,
                "face_upsample": True,
                "background_enhance": True,
                "codeformer_fidelity": 0.8
            }
        )
        codeformer_img = requests.get(codeformer_url)
        with open("codeformer_output.jpg", "wb") as f:
            f.write(codeformer_img.content)
    except Exception:
        print("CodeFormer failed — returning GFPGAN result.")
        return gfpgan_img.content

    # Шаг 3 — IDNBeauty
    try:
        beauty_url = replicate.run(
            "torrikabe-ai/idnbeauty:5f994656b3b88df2e21a3cf0a81371d66bd6ff45171f3e5618bb314bdc8b64b1",
            input={
                "image": open("codeformer_output.jpg", "rb"),
                "prompt": "A high-quality realistic selfie with smooth glowing skin, brightened whites of the eyes, softly enhanced eyelashes, subtly defined lips, and elegant facial tone — like a professional beauty filter. No changes to face structure or angle.",
                "model": "dev",
                "guidance_scale": 2,
                "prompt_strength": 0.61,
                "num_inference_steps": 28,
                "output_format": "png",
                "output_quality": 80,
                "go_fast": False,
                "lora_scale": 0.94,
                "extra_lora_scale": 0.22
            }
        )
        beauty_img = requests.get(str(beauty_url[0]))
        final_image = Image.open(io.BytesIO(beauty_img.content)).convert("RGB")
    except Exception as e:
        print(f"IDNBeauty failed: {e} — returning CodeFormer result.")
        final_image = Image.open("codeformer_output.jpg").convert("RGB")

    # Восстановление ориентации
    if orientation == 3:
        final_image = final_image.rotate(180, expand=True)
    elif orientation == 6:
        final_image = final_image.rotate(270, expand=True)
    elif orientation == 8:
        final_image = final_image.rotate(90, expand=True)

    # Конвертируем в байты и возвращаем
    img_bytes = io.BytesIO()
    final_image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes.read()