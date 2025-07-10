from fastapi import FastAPI
from distillbert import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import get_config
app = FastAPI()


model, tokenizer = load_modal_and_tokenizer()





@app.get("/transaction/{transaction}")
async def root(transaction):
    label = predict(transaction, model, tokenizer)
    return {"label" : f"{label}"}
