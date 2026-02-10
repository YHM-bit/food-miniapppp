import os
import telebot
from telebot import types

BOT_TOKEN = os.environ.get("BOT_TOKEN")
WEBAPP_URL = os.environ.get("WEBAPP_URL")

if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN env var.")
if not WEBAPP_URL:
    raise RuntimeError("Set WEBAPP_URL env var (your Render URL).")

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")


@bot.message_handler(commands=["start"])
def start(m: types.Message):
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(types.KeyboardButton("🍲 Open Cook Today", web_app=types.WebAppInfo(url=WEBAPP_URL)))

    bot.send_message(
        m.chat.id,
        "✅ Bot працює.\nНатисни кнопку нижче щоб відкрити Mini App:",
        reply_markup=kb
    )


if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling(skip_pending=True, timeout=30)

