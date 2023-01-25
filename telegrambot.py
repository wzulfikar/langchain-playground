import logging
import os
from telegram.ext import filters, ApplicationBuilder, MessageHandler

from src.bot_app import BotApp

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

if __name__ == '__main__':
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    application = ApplicationBuilder().token(token).build()

    bot = BotApp()

    # Handle any text message
    application.add_handler(MessageHandler(
        filters.TEXT & (~filters.COMMAND), bot.handle_text)
    )

    application.run_polling()
