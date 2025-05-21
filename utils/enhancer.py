import replicate
import os
import requests

replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

async def enhance_image(image_bytes: bytes) -> bytes:
    # Шаг 1: сохранить изображение временно
    with open("input.jpg", "wb") as f:
        f.write(image_bytes)

    # Шаг 2: вызвать модель Replicate
    output_url = replicate.run(
        "tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
        input={"img": open("input.jpg", "rb")}
    )

    # Шаг 3: скачать результат по ссылке
    response = requests.get(output_url)
    return response.content