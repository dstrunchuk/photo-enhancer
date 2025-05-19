from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.enhancer import enhance_image
from fastapi.staticfiles import StaticFiles
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from pydantic import BaseModel
import httpx
import io
import os
import telegram

# Telegram токен из Railway (environment variable)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BOT = telegram.Bot(token=TELEGRAM_TOKEN)

app = FastAPI()

# Разрешить Telegram WebApp (можно сузить до нужного origin)
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
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        enhanced = await enhance_image(contents)
        return StreamingResponse(io.BytesIO(enhanced), media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Telegram webhook
@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    print("[WEBHOOK] Получен update от Telegram:", data)  # ЛОГ

    chat_id = None

    # Пытаемся извлечь chat_id из всех возможных источников
    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
    elif "callback_query" in data:
        chat_id = data["callback_query"]["from"]["id"]
    elif "my_chat_member" in data:
        chat_id = data["my_chat_member"]["chat"]["id"]
    elif "edited_message" in data:
        chat_id = data["edited_message"]["chat"]["id"]

    if chat_id:
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

    
@app.post("/send_photo_upload")
async def send_photo_upload(file: UploadFile = File(...), chat_id: int = Form(...)):
    image_bytes = await file.read()

    try:
        await BOT.send_photo(chat_id=chat_id, photo=io.BytesIO(image_bytes))
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Ошибка Telegram"})

# Все маршруты зарегистрированы выше
app.mount("/", StaticFiles(directory="static", html=True), name="static")