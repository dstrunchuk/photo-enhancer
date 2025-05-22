import replicate
import requests
import os
from PIL import Image, ImageOps
import io
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime

# Инициализация клиента Replicate
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Инициализация распознавания лица
face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0)

# Проверка наличия лица на изображении
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

# Основная функция обработки
async def enhance_image(image_bytes: bytes) -> bytes:
    # Открываем изображение и выравниваем по EXIF
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image.save("input.jpg")

    # Проверка на наличие лица
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
                "upscale": 1,
                "face_upsample": True,
                "background_enhance": True,
                "codeformer_fidelity": 0.8
            }
        )
        codeformer_img = requests.get(codeformer_url)
        img_bytes_cf = io.BytesIO(codeformer_img.content)
        img_bytes_cf.seek(0)

        # Сохраняем выровненное изображение
        image_cf = Image.open(img_bytes_cf).convert("RGB")
        image_cf.save("codeformer_output.jpg", format="JPEG", quality=95)

    except Exception as e:
        print(f"CodeFormer failed: {e} — returning GFPGAN result.")
        return gfpgan_img.content
    
        # Сжимаем изображение, если оно слишком большое (Real-ESRGAN ограничен по памяти)
    image_for_esrgan = Image.open("codeformer_output.jpg").convert("RGB")
    if image_for_esrgan.width * image_for_esrgan.height > 2000000:
        scale = (2000000 / (image_for_esrgan.width * image_for_esrgan.height)) ** 0.5
        new_size = (int(image_for_esrgan.width * scale), int(image_for_esrgan.height * scale))
        image_for_esrgan = image_for_esrgan.resize(new_size, Image.LANCZOS)
        image_for_esrgan.save("codeformer_output_resized.jpg", format="JPEG", quality=95)
        esrgan_input_path = "codeformer_output_resized.jpg"
    else:
        esrgan_input_path = "codeformer_output.jpg"

     # Шаг 3 — DiffBIR (вместо Real-ESRGAN или IDNBeauty)
    try:
        diffbir_url = replicate.run(
            "zsxkib/diffbir:51ed1464d8bbbaca811153b051d3b09ab42f0bdeb85804ae26ba323d7a66a4ac",
            input={
                "input": open("codeformer_output.jpg", "rb"),
                "steps": 50,
                "tiled": False,
                "tile_size": 512,
                "tile_stride": 256,
                "repeat_times": 1,
                "use_guidance": False,
                "guidance_scale": 0,
                "guidance_space": "latent",
                "guidance_repeat": 5,
                "only_center_face": False,
                "guidance_time_stop": -1,
                "guidance_time_start": 1001,
                "background_upsampler": "DiffBIR",
                "face_detection_model": "retinaface_resnet50",
                "upscaling_model_type": "faces",
                "restoration_model_type": "general_scenes",
                "super_resolution_factor": 2,
                "disable_preprocess_model": False,
                "reload_restoration_model": False,
                "background_upsampler_tile": 400,
                "background_upsampler_tile_stride": 400
            }
        )
        diffbir_img = requests.get(str(diffbir_url[0]))
        final_image = Image.open(io.BytesIO(diffbir_img.content)).convert("RGB")

    except Exception as e:
        print(f"DiffBIR failed: {e} — returning CodeFormer result.")
        final_image = Image.open("codeformer_output.jpg").convert("RGB")
     

    # Финальный результат
    final_bytes = io.BytesIO()
    final_image.save(final_bytes, format="JPEG")
    final_bytes.seek(0)
    return final_bytes.read()