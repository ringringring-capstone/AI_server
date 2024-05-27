# fast.py
from fastapi import FastAPI
from pydantic import BaseModel

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, pipeline

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

#model_name = "RingRingRing/kogpt2_finetuning_reservation"
model_name = "mogoi/test_de_fin"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name,
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
model = GPT2LMHeadModel.from_pretrained(model_name)
  

app = FastAPI()
class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    response: str


@app.post("/chat")
async def chat(request: ChatRequest):
    input_sen = request.user_input
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=50)
    result = pipe(f"<s>[INST] {input_sen} [/INST]")

    generated_text = result[0]['generated_text']

    return ChatResponse(response=generated_text)