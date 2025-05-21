import replicate
import requests
import os
from PIL import Image
import io

replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

def compress_if_needed(image_path: str, output_path: str, max_size_mb=4, max_pixels=1600):
    size_mb = os.path.getsize(image_path) / 1024 / 1024
    if size_mb <= max_size_mb:
        return image_path  # ничего не делаем

    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_pixels:
        scale = max_pixels / max(img.size)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    img.save(output_path, format="JPEG", quality=90, optimize=True)
    return output_path

async def enhance_image(image_bytes: bytes) -> bytes:
    # Сохраняем оригинал
    with open("input.jpg", "wb") as f:
        f.write(image_bytes)

    input_path = compress_if_needed("input.jpg", "input_resized.jpg")

    # Шаг 1 — CodeFormer
    try:
        codeformer_url = replicate.run(
            "sczhou/codeformer:cc4956dd26fa5a7185d5660cc9100fab1b8070a1d1654a8bb5eb6d443b020bb2",
            input={
                "image": open(input_path, "rb"),
                "upscale": 1,
                "face_upsample": False,
                "background_enhance": False,
                "codeformer_fidelity": 0.1
            }
        )
        cf_response = requests.get(codeformer_url)
        with open("codeformer_output.jpg", "wb") as f:
            f.write(cf_response.content)
    except Exception:
        print("Ошибка в CodeFormer — возвращаю исходник.")
        return image_bytes

    cf_path = compress_if_needed("codeformer_output.jpg", "cf_resized.jpg")

    # Шаг 2 — IDNBeauty
    try:
        beauty_url = replicate.run(
            "torrikabe-ai/idnbeauty:5f994656b3b88df2e21a3cf0a81371d66bd6ff45171f3e5618bb314bdc8b64b1",
            input={
                "image": open(cf_path, "rb"),
                "prompt": "A high-quality realistic portrait of a beautiful person with softly glowing skin, naturally brightened eyes, delicate eyelashes, subtly emphasized eye corners, and slightly enhanced lips. The facial features remain authentic. The effect is clean, fresh, and elegant — like professional soft studio light with gentle retouch.",
                "model": "dev",
                "guidance_scale": 2,
                "prompt_strength": 0.61,
                "num_inference_steps": 28,
                "output_format": "png",
                "output_quality": 90,
                "megapixels": "1",
                "go_fast": False,
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "extra_lora_scale": 0.22,
                "lora_scale": 0.94
            }
        )
        beauty_response = requests.get(beauty_url)
        final_image = Image.open(io.BytesIO(beauty_response.content)).convert("RGB")
    except Exception:
        print("Ошибка в IDNBeauty — возвращаю CodeFormer.")
        final_image = Image.open(cf_path).convert("RGB")

    # Финальное сжатие, если > 5 МБ
    output_io = io.BytesIO()
    final_image.save(output_io, format="JPEG", quality=95, optimize=True)
    if output_io.getbuffer().nbytes > 5 * 1024 * 1024:
        print("Финальный файл > 5 МБ — пересохраняем.")
        output_io = io.BytesIO()
        final_image.save(output_io, format="JPEG", quality=90, optimize=True)

    output_io.seek(0)
    return output_io.read()