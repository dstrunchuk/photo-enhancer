import replicate
import requests
import os
from PIL import Image
import io

replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

def compress_and_resize(image_path: str, output_path: str, max_size=1600):
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)
    image.save(output_path, format="JPEG", quality=90, optimize=True)

async def enhance_image(image_bytes: bytes) -> bytes:
    with open("input.jpg", "wb") as f:
        f.write(image_bytes)

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
                "codeformer_fidelity": 0.1
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
                "prompt": "A high-quality realistic portrait of a beautiful person with softly glowing skin, naturally brightened eyes, delicate eyelashes, subtly emphasized eye corners, and slightly enhanced lips. The facial features remain authentic. The effect is clean, fresh, and elegant — like professional soft studio light with gentle retouch.",
                "model": "dev",
                "guidance_scale": 2,
                "prompt_strength": 0.61,
                "num_inference_steps": 28,
                "output_format": "png",
                "output_quality": 80,
                "megapixels": "1",
                "go_fast": False,
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "extra_lora_scale": 0.22,
                "lora_scale": 0.94
            }
        )
        beauty_img = requests.get(beauty_url)
        final_image = Image.open(io.BytesIO(beauty_img.content)).convert("RGB")
    except Exception:
        print("IDNBeauty failed — returning CodeFormer result.")
        final_image = Image.open("codeformer_resized.jpg").convert("RGB")

    # Возвращаем уже готовое изображение без финального сжатия
    img_bytes = io.BytesIO()
    final_image.save(img_bytes, format="JPEG")  # без quality/optimize
    img_bytes.seek(0)
    return img_bytes.read()  