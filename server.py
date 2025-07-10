from fastapi import FastAPI
from

app = FastAPI()

def setup_model():

@app.get("/fake-transcation-detector")
async def root(transaction):
    return {"message" : "hello world"}
