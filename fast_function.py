import random
import pandas as pd

# 모델 및 토크나이저 관련 라이브러리
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import random



def LoadData():
    data_path = "data/delivery/start_sentence.csv"
    situ_data = pd.read_csv(data_path)
    situation = situ_data['situ'].to_list()
    return situation


# random 추출
def Random_situ(action):
    data = pd.read_csv("start_sentence.csv")
    data_len = len(data['situ'].to_list())

    ran_dataset = [(data['situ'][i], data['start_sen'][i]) for i in range(data_len)]
    '''
    ran_dataset = []
    for s in range(data_len):
        situation = data['situ'][s]
        start_sentence = data['start_sen'][s]
        ran_dataset.append([situation, start_sentence])'''

    if action == 'Start':
        random_choice = random.choice(ran_dataset)
        random_data = random_choice
        return random_data
    elif action == 'Cancel':
        # random 값 다 사라진 경우
        if not ran_dataset:
            raise HTTPException(status_code=400, detail="No more available situations")
        
        # 리스트에서 지우고
        ran_dataset.remove(random_choice)
        # 다시 random 돌리기
        random_data = random.choice(ran_dataset)
        return random_data
        


# 사용자 입력 기반으로 응답 생성
tokenizer = PreTrainedTokenizerFast.from_pretrained("kogpt2_fine_model(05.14).tar")
model_path = "model/kogpt2_fine_model(05.14)"
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
def generate_response(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

# fast api을 통해 입출력 시 딱히 필요 없음
# 모델이랑 입출력으로 대화 나누는 부분(ChatBot())
'''
def ChatBot():
    # 챗봇
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            #print("Chatbot: Goodbye!")
            break
        else:
            response = generate_response(tokenizer, device, model, user_input, max_length=50)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            #print("Chatbot:", response)

if __name__ == '__main__':
    ChatBot()
'''