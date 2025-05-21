import replicate
import requests
import os
from PIL import Image
import io

replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

async def enhance_image(image_bytes: bytes) -> bytes:
    # Сохраняем оригинальное фото
    with open("input.jpg", "wb") as f:
        f.write(image_bytes)

    # --- Шаг 1: GFPGAN ---
    gfpgan_url = replicate.run(
        "tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
        input={"img": open("input.jpg", "rb")}
    )

    # Скачиваем результат GFPGAN
    gfpgan_img_response = requests.get(gfpgan_url)
    with open("gfpgan_output.jpg", "wb") as f:
        f.write(gfpgan_img_response.content)

    # --- Шаг 2: CodeFormer ---
    codeformer_url = replicate.run(
        "sczhou/codeformer",
        input={
            "image": open("gfpgan_output.jpg", "rb"),
            "fidelity": 1.0
        }
    )

    # Скачиваем результат CodeFormer
    codeformer_img_response = requests.get(codeformer_url)
    final_image = Image.open(io.BytesIO(codeformer_img_response.content)).convert("RGB")

    # --- Сжатие до ~5 МБ ---
    output_buffer = io.BytesIO()
    final_image.save(output_buffer, format="JPEG", quality=95, optimize=True)
    output_buffer.seek(0)

    return output_buffer.read()