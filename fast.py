from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import os
import signal

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

app = FastAPI()

# Hugging Face에서 모델과 토크나이저를 불러옵니다.
#'mogoi/kogpt2_finetuning_delivery'
U_TKN = '<usr>'
S_TKN = '<sys>'
EOS = '</s>'
MASK = '<unused0>'
PAD = '<pad>'
SENT = '<unused1>'
# skt/kogpt2-base-v2
tokenizer = PreTrainedTokenizerFast.from_pretrained("mogoi/kogpt2_finetuning_delivery",
            eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
model = GPT2LMHeadModel.from_pretrained('mogoi/kogpt2_finetuning_reservation')
'''
checkpoint_path = '02_reservation/last.ckpt'
state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# 모델의 가중치 업데이트
model.load_state_dict(state_dict)
'''

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 'quit' 명령어를 처리합니다.
        if request.user_input.lower() == 'quit':
            # 서버 종료 메시지를 반환합니다.
            response_text = "연습을 종료합니다. 이용해 주셔서 감사합니다."
            # 서버를 종료하는 비동기 작업을 스케줄링합니다.
            asyncio.create_task(shutdown_server())
            return {"response": response_text}
        
        # 입력된 문장을 토큰화합니다.
        input_ids = tokenizer.encode(request.user_input, return_tensors='pt')
        
        # 모델을 사용하여 응답을 생성합니다.
        with torch.no_grad():
            output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        
        # 토큰을 텍스트로 변환합니다.
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def shutdown_server():
    """서버를 종료하는 함수"""
    await asyncio.sleep(1)  # 응답을 클라이언트에 전송할 시간을 줍니다.
    import os
    import signal
    os.kill(os.getpid(), signal.SIGINT)