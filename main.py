from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
import httpx
import io
import os
import telegram
import traceback
from utils.enhancer import enhance_image

# Telegram и OpenAI токены
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOT = telegram.Bot(token=TELEGRAM_TOKEN)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PhotoData(BaseModel):
    chat_id: int
    photo_url: str



# Эндпоинт загрузки фото
@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...)
):
    try:
        contents = await file.read()
        result = await enhance_image(contents)
        return StreamingResponse(io.BytesIO(result), media_type="image/jpeg")
    except Exception as e:
        print("ERROR DURING IMAGE PROCESSING:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Telegram webhook
@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    try:
        message = data.get("message", {})
        chat_id = message.get("chat", {}).get("id")

        if not chat_id:
            print("[WEBHOOK] ❌ Chat ID не найден")
            return {"ok": False}

        print(f"[WEBHOOK] Chat ID: {chat_id}")

        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": "Привет, я улучшаю фотографии с помощью нейросетей — в один клик!",
                    "reply_markup": {
                        "inline_keyboard": [[
                            {
                                "text": "ОТКРЫТЬ",
                                "web_app": {
                                    "url": "https://photo-enhancer-production.up.railway.app"
                                }
                            }
                        ]]
                    }
                }
            )

        return {"ok": True}

    except Exception as e:
        print("[WEBHOOK ERROR]", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/send_photo_upload")
async def send_photo_upload(file: UploadFile = File(...), chat_id: int = Form(...)):
    try:
        image_bytes = await file.read()

        await run_in_threadpool(
            BOT.send_document,
            chat_id=chat_id,
            document=io.BytesIO(image_bytes),
            filename="uluchshennoe_foto.jpg",
            caption="Ваше улучшенное фото (без сжатия)"
        )

        return {"ok": True}
    except Exception as e:
        print("[SEND ERROR]", str(e))
        return JSONResponse(status_code=500, content={"error": "Ошибка Telegram"})

# Подключение frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")