import replicate
import requests
import os
from PIL import Image
import io

replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

async def enhance_image(image_bytes: bytes) -> bytes:
    # Сохраняем исходный файл
    with open("input.jpg", "wb") as f:
        f.write(image_bytes)

    # --- Шаг 1: GFPGAN ---
    gfpgan_url = replicate.run(
        "tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
        input={"img": open("input.jpg", "rb")}
    )
    gfpgan_img_response = requests.get(gfpgan_url)
    with open("gfpgan_output.jpg", "wb") as f:
        f.write(gfpgan_img_response.content)

    # --- Шаг 2: CodeFormer ---
    codeformer_url = replicate.run(
        "sczhou/codeformer:cc4956dd26fa5a7185d5660cc9100fab1b8070a1d1654a8bb5eb6d443b020bb2",
        input={
            "image": open("gfpgan_output.jpg", "rb"),
            "upscale": 2,
            "face_upsample": True,
            "background_enhance": True,
            "codeformer_fidelity": 0.1
        }
    )
    codeformer_response = requests.get(codeformer_url)
    with open("codeformer_output.jpg", "wb") as f:
        f.write(codeformer_response.content)

    # --- Шаг 3: IDNBeauty ---
    beauty_url = replicate.run(
        "torrikabe-ai/idnbeauty",
        input={"image": open("codeformer_output.jpg", "rb")}
    )
    beauty_response = requests.get(beauty_url)

    # --- Финальная конвертация и сжатие ---
    final_image = Image.open(io.BytesIO(beauty_response.content)).convert("RGB")
    buffer = io.BytesIO()
    final_image.save(buffer, format="JPEG", quality=95, optimize=True)
    buffer.seek(0)

    return buffer.read()