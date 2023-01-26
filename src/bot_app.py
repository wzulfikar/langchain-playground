import logging
import os
from openai.error import OpenAIError
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

DEFAULT_PROMPT_TEMPLATE = """Assistant is a large language model trained by OpenAI.
  Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
  Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
  Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
  {history}
  Human: {human_input}
  Assistant:"""
DEFAULT_START_MESSAGE = "Hello, I'm an AI-powered chatbot 😊\nSend me a message and I'll try to answer it."


class BotApp:
    def __init__(self):
        # You can adjust the template as needed. Reload the bot to apply changes.
        self.chains = {}
        self.prompt_template = os.environ.get(
            "PROMPT_TEMPLATE", DEFAULT_PROMPT_TEMPLATE)
        self.is_verbose = os.environ.get("VERBOSE") == "1"

    async def handle_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await ctx.bot.send_message(chat_id=update.effective_chat.id,
                                   text=os.environ.get("START_MESSAGE", DEFAULT_START_MESSAGE))

    async def handle_text(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = update.message.text

        chat_id = update.message.chat.id
        sender_name = update.message.from_user.username or update.message.from_user.first_name
        is_group = update.message.chat.type == "group"
        is_reply = update.message.reply_to_message

        # Prepare session for the user
        chain = self.chains.get(chat_id)
        if not chain:
            # chain is unique per chat
            chain = create_chain(self.prompt_template, self.is_verbose)
            self.chains[chat_id] = chain
            logging.info("  new chain created: %d", chat_id)

        if text.lower() == "ping":
            await ctx.bot.send_message(chat_id=update.effective_chat.id,
                                       text="pong!")
        else:
            start_time = datetime.datetime.now()

            if is_group:
                # Skip if message is from group but doesn't have mention
                if not text.startswith(f"@{ctx.bot.username}"):
                    return
                # Remove mention from the text
                text = text[len(f"@{ctx.bot.username}") + 1:]

            # Use reply text as prompt
            if is_reply:
                text = f"{text}: {update.message.reply_to_message.text}"

            logging.info("[message] new %smessage from %s",
                         "group " if is_group else "", sender_name)

            with get_openai_callback() as cb:
                await ctx.bot.send_chat_action(chat_id=update.effective_chat.id,
                                               action="typing")
                logging.info("  predicting...")

                try:
                    output = chain.predict(human_input=text)
                    bot_reply = output
                except OpenAIError as e:
                    logging.error("OpenAIError: %s", e)
                    bot_reply = "Sorry, I seem to have some issues 😕\nPlease try again later."

                await ctx.bot.send_message(chat_id=update.effective_chat.id,
                                           text=bot_reply)

                logging.info("  human_input: %s", text)
                logging.info("  output: %s", bot_reply)
                logging.info("  total_tokens: %d", cb.total_tokens)
                logging.info("  time_elapsed_ms: %d ms",
                             (datetime.datetime.now() - start_time).microseconds / 1000)
