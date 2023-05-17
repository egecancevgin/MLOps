from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version

app = FastAPI()

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    language: str

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    language = predict_pipeline(payload.text)
    return {"language": language}




from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}



