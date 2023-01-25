import logging
import os
from telegram import Update
from telegram.ext import ContextTypes
from src.llm.chain import create_chain

# TODO: hide langchain from telegrambot.py
import datetime
from langchain.callbacks import get_openai_callback

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class BotApp:
    def __init__(self, prompt_template):
        self.chain = create_chain(
            prompt_template, os.environ.get("VERBOSE") == "1")

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle simple message 'ping' to act as a health check"""
        text = update.message.text

        # create variable for the name of the sender
        sender_name = update.message.from_user.username or update.message.from_user.first_name

        if text == "ping":
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text="pong!")
        else:
            logging.info("[message] new message from %s", sender_name)
            start_time = datetime.datetime.now()

            with get_openai_callback() as cb:
                logging.info("  predicting...")
                output = self.chain.predict(human_input=text)
                bot_reply = output.strip()
                await context.bot.send_message(chat_id=update.effective_chat.id,
                                               text=bot_reply)

                logging.info("  human_input: %s", text)
                logging.info("  output: %s", bot_reply)
                logging.info("  total_tokens: %d", cb.total_tokens)
                logging.info("  time_elapsed_ms: %d ms",
                             (datetime.datetime.now() - start_time).microseconds / 1000)
