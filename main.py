from fastapi import FastAPI, UploadFile, File, Request
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
    update = telegram.Update.de_json(data, BOT)

    if update.message:
        chat_id = update.message.chat_id
        if update.message:
            chat_id = update.message.chat_id
            BOT.send_message(
                chat_id=chat_id,
                text="Привет, я улучшаю фотографии с помощью нейросетей — в один клик!",
                reply_markup = InlineKeyboardMarkup([[
                    InlineKeyboardButton(
                        text="ОТКРЫТЬ",
                        web_app=WebAppInfo(url="https://photo-enhancer-production.up.railway.app")
                    )
                ]])
            )

    return {"ok": True}

@app.post("/send_photo")
async def send_photo(data: PhotoData):
    async with httpx.AsyncClient() as client:
        tg_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        response = await client.post(
            tg_url,
            data={
                "chat_id": data.chat_id,
                "photo": data.photo_url,
                "caption": "Ваша улучшенная фотография"
            }
        )
    return {"ok": response.status_code == 200}

# Все маршруты зарегистрированы выше
app.mount("/", StaticFiles(directory="static", html=True), name="static")