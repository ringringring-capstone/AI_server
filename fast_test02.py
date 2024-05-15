from fastapi import FastAPI, Request
from random import choice
from pydantic import BaseModel

# 파이썬 파일 불러오기
from fast_function import LoadData, run_model, Ran_start_situ


app = FastAPI()


# 문장 입출력
class Conversation(BaseModel):
    sentences: list[str]=[]

@app.get("/practice/{item_id}")
async def create_sub_title(item_id: str):  # str : 배달, 예약, 상담 중 하나 들어감 
    # 이름에 맞는 모델 불러오기.
    #소주제 선택하고 return 해서
    return {"item_id": item_id}

@app.get("/mission")
async def create_mission():  # str : 배달, 예약, 상담 중 하나 들어감 
    # 배달 예약 상담 중 하나 random으로 부르기

# 모델 이름을 기반으로 해당 모델 함수를 실행하는 엔드포인트
@app.post("/process_model/")
async def process_model(request: Request):
    data = await request.json()
    model_name = data.get("model")
    input_data = data.get("input_data")

    if model_name in ["delivery", "reservation", "consulting"]:
        output = run_model(model_name, input_data)
    else:
        output = {"error": "Invalid model name."}

    return {"output": output}

'''
# 문장 입출력
class Conversation(BaseModel):
    sentences: list[str]=[]

@app.get("/practice/{item_id}")
async def create_sub_title(item_id: str):  # str : 배달, 예약, 상담 중 하나 들어감 
    # 이름에 맞는 모델 불러오기.
    #소주제 선택하고 return 해서
    return {"item_id": item_id}


@app.get("/mission")
async def create_mission():  # str : 배달, 예약, 상담 중 하나 들어감 
    # 배달 예약 상담 중 하나 random으로 부르기

    return {}

# 대화 생성
@app.post("/kogpt2-test")
async def create_sentence(sentence: Conversation):
    random_situation = ChatBot()

    user_input = sentence.sentence


    for i in sentence:
        print (i)
    return "new sentence"
'''




# 주제 랜덤 생성
# 주제 random 생성

@app.post("/start_situ/")
async def ran_start_situ(action: str, selected_value: str = None):
    global ran_dataset
    ran_dataset = Ran_start_situ()
    
    # action이 start이면 ran_dataset에서 랜덤하게 값을 추출하여 반환
    if action == "start":
        selected_data = choice(ran_dataset)
        return {"situ": selected_data[0], "start_sen": selected_data[1]}
    
    # action이 start이 아닌 경우
    elif action == "select":        
        # 선택한 값을 제외한 나머지 데이터를 추출하여 ran_dataset 업데이트
        ran_dataset = [data for data in ran_dataset if data[0] != selected_value]
        
        # ran_dataset이 비어 있는지 확인
        if not ran_dataset:
            return {"message": "No more options available."}
        
        # 남은 데이터 중에서 랜덤하게 값을 추출하여 반환
        selected_data = choice(ran_dataset)
        return {"situ": selected_data[0], "start_sen": selected_data[1]}