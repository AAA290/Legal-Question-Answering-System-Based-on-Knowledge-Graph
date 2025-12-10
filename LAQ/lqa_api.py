"""
基于FastAPI框架构建的智能法律服务平台核心接口文件，集成了法律问答、罪名预测、实体识别以及问题分类三大核心功能模块。
系统采用RESTful API架构设计，支持高并发异步处理，提供标准化JSON数据交互接口
为网页提供服务
"""
from fastapi.testclient import TestClient
from fastapi import FastAPI,Body
from chatbot import ChatBot
from use_crime_prediction import crimes_prediction
from question_classifier import QuestionClassifier
from graph_search import GraphSearch

app = FastAPI()
@app.post("/api/qa")
async def answer(question: str = Body(...,embed = True)):
    bot = ChatBot()
    return {"answer": bot.chat_main(question)}

# @app.post("/api/qa_com")
# async def answer_com(question: str = Body(...,embed = True)):
#     bot = ChatBot()
#     return {"answer": bot.chat_main(question)}

@app.post("/api/crimes_prediction")
async def predict(fact:  str = Body(...,embed = True)):
    return {"result": crimes_prediction(fact)}

@app.post("/api/ner")
async def ner_classifier(question:  str = Body(...,embed = True)):
    qc = QuestionClassifier()
    return {"result": qc.classify(question)}

@app.post("/api/flowchart")
async def answer(question: str = Body(...,embed = True)):
    bot = ChatBot()
    qc = QuestionClassifier()
    entity_type = qc.classify(question)
    gs = GraphSearch()
    ty_cy_entity = gs.Cypher_search(entity_type)
    cleaned_cyp = ' '.join(ty_cy_entity["cyp"].split())
    type_map = {
        '1': '[crime]构成要件',
        '2': '[crime]的[Punishment]',
        '3': '[Law] 内容',
        '4': '[crime]相关的[Law]',
        '5': '[Law]相关的[crime]',
        '6': '[crime] 定义',
        '7': '[crime]包含的[crime]'
    }
    # 构建模板映射字典
    template_map = {
        '1': '构成要件为',
        '2': '刑罚包括',
        '3': '内容有',
        '4': '相关法条有',
        '5': '涉及的罪名有',
        '6': '概念为',
        '7': '包含的罪名有'
    }
    # 统一赋值逻辑
    temp = f'{ty_cy_entity["entity"]}的{template_map[ty_cy_entity["type"]]}: '
    result = {"entity":entity_type[0],"type":type_map[entity_type[1]],"cypher":cleaned_cyp,"temp":temp,"answer":bot.chat_main(question)}
    return result

# 使用 TestClient 进行测试
client = TestClient(app)

def test_answer():
    # 发送 POST 请求，参数通过 JSON 传入
    response = client.post("/api/qa", json={"question": "盗窃罪怎么判刑？"})
    print("返回结果：", response.json())

def test_predict():
    response = client.post("/api/crimes_prediction", json={"fact": "被告先盗窃商店财物，后持刀威胁店员构成抢劫"})
    print("返回结果：", response.json())

def test_ner():
    response = client.post("/api/ner", json={"question": "盗窃罪怎么判刑？"})
    print("返回结果：", response.json())

def test_flowchart():
    response = client.post("/api/flowchart", json={"question": "盗窃罪怎么判刑？"})
    print("返回结果：", response.json())

if __name__ == "__main__":
    # test_answer()
    # test_predict()
    # test_ner()
    test_flowchart()