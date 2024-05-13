from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

# 모델 및 토크나이저 관련 라이브러리
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import random
import pandas as pd
# 파이썬 파일 불러오기
from chat import Random_situ, ChatBot

app = FastAPI()

@app.get("/")
def root():
    return {"message":"Hello World"}


@app.get("/home")
def root():
    return {"message":"home"}



# 토크나이저
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

tokenizer = PreTrainedTokenizerFast.from_pretrained("kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)
model_path = "model/fine_tuned_model"
model = GPT2LMHeadModel.from_pretrained(model_path)

# 주제 random 생성
class SituationRequest(BaseModel):
    situation: str

situations_df = pd.read_csv("situation_data.csv")
excluded_situations = []

@app.get("/start")
def start_exercise():
    situation = random_situ()
    return {"situation": situation}

@app.get("/select/{situation_id}")
def select_situation(situation_id: int):
    global excluded_situations
    if situation_id < 0 or situation_id >= len(situations_df):
        return {"error": "Invalid situation ID"}
    elif situation_id in excluded_situations:
        return {"message": "This situation has been excluded. Please try again."}
    else:
        situation = situations_df.iloc[situation_id]["situ"]
        return {"selected_situation": situation}

@app.get("/cancel")
def cancel_selection():
    global excluded_situations
    excluded_situations.append(random.choice(range(len(situations_df))))
    situation = random_situ()
    return {"situation": situation}

def random_situ():
    global excluded_situations
    available_situations = [i for i in range(len(situations_df)) if i not in excluded_situations]
    if len(available_situations) == 0:
        excluded_situations = []  # Reset excluded situations if all have been excluded
        available_situations = [i for i in range(len(situations_df))]
    random_index = random.choice(available_situations)
    situation = situations_df.iloc[random_index]["situ"]
    return situation


# 문장 입출력
class Conversation(BaseModel):
    sentences: list[str]=[]

@app.post("/kogpt2-test")
async def create_sentence(sentence: Conversation):
    random_situation = ChatBot()

    user_input = sentence.sentence


    for i in sentence:
        print (i)
    return "new sentence"