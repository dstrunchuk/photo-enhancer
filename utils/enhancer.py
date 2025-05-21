from PIL import Image
import io
import replicate
import requests
import os

replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

async def enhance_image(image_bytes: bytes) -> bytes:
    # Сохраняем временный оригинал
    with open("input.jpg", "wb") as f:
        f.write(image_bytes)

    # Запускаем Replicate модель
    output_url = replicate.run(
        "tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
        input={"img": open("input.jpg", "rb")}
    )

    # Скачиваем результат
    response = requests.get(output_url)
    image = Image.open(io.BytesIO(response.content)).convert("RGB")

    # Сохраняем с высоким качеством — нацелено на ~5 МБ
    compressed_io = io.BytesIO()

    # Качество 95 = почти без потерь, но большой размер
    image.save(compressed_io, format="JPEG", quality=95, optimize=True)
    compressed_io.seek(0)

    # Опционально: контроль размера, если нужно точнее
    # print(f"Final size: {len(compressed_io.getvalue()) / 1024 / 1024:.2f} MB")

    return compressed_io.read()