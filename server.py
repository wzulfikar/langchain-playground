from typing import Union
from fastapi import FastAPI
from src.handler import llm

app = FastAPI()


@app.get("/ping")
def read_ping():
    return {"ok": True, "message": "pong!"}


@app.get("/api/llm")
def predict_llm(text: str):
    return llm.predict(text)
