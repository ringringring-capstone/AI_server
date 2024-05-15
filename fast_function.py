import random
import pandas as pd

# 모델 및 토크나이저 관련 라이브러리
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import random
import importlib.util

# 사용자 입력 기반으로 응답 생성
tokenizer = PreTrainedTokenizerFast.from_pretrained("model/KoGPT2_checkpoint_(05.14).tar")
model_path = "model/{kogpt2_fine_model(05.14)}"
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def run_model(model_name, input_data):
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
    model_path = "model/{model_name}"
    model = GPT2LMHeadModel.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 모델이 저장된 폴더의 경로
    model_folder_path = f"model/{model_name}/"
    # 모델 파일의 경로
    model_file_path = f"{model_folder_path}{model_name}.py"

    input_ids = tokenizer.encode(input_data, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    # 생성된 응답 출력
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.replace(input_ids, "").strip()
    
    return response

    try:
        # 모듈을 동적으로 로드
        spec = importlib.util.spec_from_file_location(model_name, model_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 모델 함수 호출
        output = module.run_model(input_data)
        return output
    except Exception as e:
        return f"Error occurred while running model {model_name}: {str(e)}"


def LoadData():
    data_path = "data/delivery/start_sentence.csv"
    situ_data = pd.read_csv(data_path)
    situation = situ_data['situ'].to_list()
    return situation

# random 추출
def Ran_start_situ():
    data = pd.read_csv("data/delivery/start_sentence.csv")
    data_len = len(data['situ'].to_list())

    ran_dataset = [(data['situ'][i], data['start_sen'][i]) for i in range(data_len)]

    return ran_dataset
'''
# random 추출
def Random_situ(action):
    data = pd.read_csv("start_sentence.csv")
    data_len = len(data['situ'].to_list())

    ran_dataset = [(data['situ'][i], data['start_sen'][i]) for i in range(data_len)]

    if action == 'Start':
        random_choice = random.choice(ran_dataset)
        
        while True:
            return random_data
    
            if action == 'Cancel':
                # random 값 다 사라진 경우
                if not ran_dataset:
                    raise HTTPException(status_code=400, detail="No more available situations")
                
                # 리스트에서 지우고
                ran_dataset.remove(random_choice)
                # 다시 random 돌리기
                random_data = random.choice(ran_dataset)
                return random_data
        



'''
    
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