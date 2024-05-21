from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# FastAPI 애플리케이션 초기화
app = FastAPI()

# 모델과 토크나이저 로드 (Hugging Face Model Hub에서)
model_name = "RingRingRing/kogpt2_finetuning_reservation"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 요청 데이터 모델 정의
class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    response: str


# 대화 함수 정의
def generate_response(user_input: str) -> str:
    U_TKN = '<usr>'
    S_TKN = '<sys>'
    SENT = '<unused1>'
    EOS = '</s>'
    
    input_ids = tokenizer.encode(U_TKN + user_input + SENT + S_TKN, return_tensors='pt')
    with torch.no_grad():
        pred = model(input_ids)
        gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred.logits, dim=-1).squeeze().tolist())
        response = ''.join(gen).replace('▁', ' ').replace(EOS, '').strip()
    return response

# POST 엔드포인트 정의
@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    user_input = chat_request.user_input
    response = generate_response(user_input)
    return ChatResponse(response=response)