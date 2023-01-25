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
    def __init__(self):
        # You can adjust the template as needed. Reload the bot to apply changes.
        self.template = """Assistant is a large language model trained by OpenAI.
  Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
  Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
  Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
  {history}
  Human: {human_input}
  Assistant:"""
        self.chains = {}
        self.is_verbose = os.environ.get("VERBOSE") == "1"

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = update.message.text

        sender_id = update.message.from_user.id
        sender_name = update.message.from_user.username or update.message.from_user.first_name
        logging.info("[message] new message from %s", sender_name)

        # Prepare session for the user
        chain = self.chains.get(sender_id)
        if not chain:
            chain = create_chain(self.template, self.is_verbose)
            self.chains[sender_id] = chain
            logging.info("  new chain created: %d", sender_id)

        if text == "ping":
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text="pong!")
        else:
            start_time = datetime.datetime.now()

            with get_openai_callback() as cb:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id,
                                                   action="typing")
                logging.info("  predicting...")

                output = chain.predict(human_input=text)
                bot_reply = output.strip()
                await context.bot.send_message(chat_id=update.effective_chat.id,
                                               text=bot_reply)

                logging.info("  human_input: %s", text)
                logging.info("  output: %s", bot_reply)
                logging.info("  total_tokens: %d", cb.total_tokens)
                logging.info("  time_elapsed_ms: %d ms",
                             (datetime.datetime.now() - start_time).microseconds / 1000)
