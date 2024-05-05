from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel


app = FastAPI()

@app.get("/")
def root():
    return {"message":"Hello World"}


@app.get("/home")
def root():
    return {"message":"home"}




class Conversation(BaseModel):
    sentences: list[str]=[]





@app.post("/kogpt2-test")
async def create_sentence(sentence: Conversation):
    for i in sentence:
        print (i)
    return "new sentence"