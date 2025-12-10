import logging
from question_classifier import *
from graph_search import *
from answer_generator import *
import socket

logging.basicConfig(filename="./output/chatbot_test.log",  # 将日志保存到filename文件中
                    filemode='w',  # 写入的模式
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    level=logging.DEBUG)  # DEBUG及以上的日志信息都会显示

def check_port(host, port):
    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

class ChatBot:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.graph_searcher = GraphSearch()
        self.answer_generator = AnswerGenerator()

    def chat_main(self,question):
        default_answer = '无法理解您的问题，请咨询专业律师。'
        if question == '结束服务':
            return '感谢使用，再见！'
        logging.info('1.现在开始进行实体识别与分类环节...')
        entity,type = result =  self.classifier.classify(question)
        logging.debug(f'entity: {entity}\ntype: {type}')
        if not entity or not type:
            return default_answer
        logging.info('2.现在开始进行对应产生cypher查询语句环节...')
        res_cyp = self.graph_searcher.Cypher_search(result)
        logging.debug(f'res_cyp: {res_cyp}')
        logging.info('3.现在开始进行产生答案环节...')
        answer = self.answer_generator.generate_main(res_cyp)
        if not answer:
            return default_answer
        return answer

if __name__ == '__main__':
    handler = ChatBot()
    while 1:
        question = input('用户:')
        if question == '结束服务':
            print('bot: 感谢使用，再见！')
            break
        answer = handler.chat_main(question)
        print('bot:', answer)