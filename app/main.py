import sys
sys.path.append('../')

from fastapi import FastAPI
from inference.inference import Inference

app = FastAPI()
inference = Inference()
inference.load_model()

@app.get("/")
def health_check():
    return {"status": "OK"}

@app.get("/info")
def info():
    return {
        "model_name": inference.config.hf_model_name,
        "context_length": inference.config.context_length
    }

@app.get("/predict")
def predict(question: str, context: str):
    start_token_idx, end_token_idx, answer = inference.predict(question, context)
    return {
        "start_token_idx": start_token_idx,
        "end_token_idx": end_token_idx,
        "answer": answer
    }
