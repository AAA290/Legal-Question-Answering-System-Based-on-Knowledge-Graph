from py2neo import Graph

# 配置Neo4j连接
URL = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

class AnswerGenerator:
    def __init__(self):
        self.graph = Graph(URL,
                           auth=(USER,PASSWORD),
                           max_age=3000,
                           max_size=100)

    def generate_main(self,res):
        question_type = res['type']
        query = res['cyp']
        entity = res['entity']
        answers = self.graph.run(query).data()
        final_answer = self.answer_prettify(question_type,entity, answers)
        return final_answer

    def answer_prettify(self, query_type,entity, answers):
        final_answer = []
        if not answers:
            return "无法理解您的问题，请咨询专业律师。"

        # Type 1: [crime]构成要件
        ## 可以优化为根据type一起显示，一个type的显示在一起
        if query_type == '1':
            final_answer = f'{entity}的构成要件为：\n'+''.join('{0}：{1}\n'.format(answer['f.type'],answer['f.content']) for answer in answers)

        # Type 2: [crime]的[Punishment]
        elif query_type == '2':
            final_answer = f'{entity}的刑罚包括：\n'+''.join('{0}\n'.format(answer['p.content']) for answer in answers)

        # Type 3: [Law]内容
        elif query_type == '3':
            final_answer = f'{entity}的内容有：\n'+''.join('{0}\n'.format(answer['l.content']) for answer in answers)

        # Type 4: [crime]相关的[Law]
        elif query_type == '4':
            final_answer = f'{entity}的相关法条有：\n'+''.join('{0}\n'.format(answer['l.content']) for answer in answers)

        # Type 5: [Law]对应的[crime]
        elif query_type == '5':
            final_answer = f'{entity}涉及的罪名有：\n'+''.join('{0}\n'.format(answer['c.name']) for answer in answers)

        # Type 6: [crime]定义
        elif query_type == '6':
            final_answer = f'{entity}的概念为：\n'+''.join('{0}\n'.format(answer['c.concept']) for answer in answers)

        # Type 7: [big crime]包含的[small crime]
        elif query_type == '7':
            final_answer = f'{entity}包含的罪名有：\n'+''.join('{0}\n'.format(answer['c.name']) for answer in answers)

        return final_answer

             