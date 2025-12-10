"""
源数据集为CAIL2019
只提取所有criminal domain的裁决文书
用于进行实体标注，作为后续NER模块的训练数据
"""

import json

data_file = "../data/context.json"
save_file = "../output/facts.txt"
def clean_data():
    with open(data_file,'r',encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    with open(save_file,'w', encoding='utf-8') as fs:
        for item in data:
            for case in item['data']:
                if case['domain'] == 'criminal':
                    for paragraph in case['paragraphs']:
                        fs.write(paragraph['context'] + '\n')

if __name__ == "__main__":
    clean_data()