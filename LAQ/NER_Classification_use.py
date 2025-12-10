"""
推理脚本：加载训练好的联合模型，实现对单条问句的实体识别与关系分类

功能：
    1. 初始化：加载 tokenizer 与 JointEntityRelationModel 权重。
    2. classify(question)：
        a. 对输入问句按字符分词并编码；
        b. 前向推理，获得实体标签序列与关系 logits；
        c. CRF 解码得到实体边界与类型，重组为 (type, text) 列表；
        d. 关系分类取 argmax，输出模式编号字符串。
    3. 返回值：entities 列表与 pattern（1–7 字符串），可直接用于后续 Cypher 查询。

使用示例：
    from NER_Classification_use import QuestionClassifier
    qc = QuestionClassifier()
    entities, pattern = qc.classify("盗窃罪怎么判刑")
"""

import torch
from transformers import AutoTokenizer
from NER_Classification_model import JointEntityRelationModel
import logging
logging.basicConfig(filename='./output/log.txt',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                    level=logging.INFO)

class NER_CLS:
    def __init__(self, model_dir='model/electra_joint_model'):
        # 加载 tokenizer 与模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = JointEntityRelationModel(
            encoder_name=model_dir,
            num_entity_tags=7,
            num_relations=7
        )
        self.model.load_state_dict(torch.load(f'{model_dir}/pytorch_model.bin',
                                              map_location='cpu'), strict=False)
        self.model.eval()

    def classify(self, question):
        # 1. 编码
        enc = self.tokenizer(list(question),
                             is_split_into_words=True,
                             return_tensors='pt',
                             padding='max_length',
                             truncation=True,
                             max_length=40)

        # print(">>> tokens:", self.tokenizer.convert_ids_to_tokens(enc.input_ids[0]))
        # print(">>> attention_mask:", enc.attention_mask[0].tolist())
        # print(">>> input_ids   :", enc.input_ids[0].tolist())

        out = self.model(input_ids=enc.input_ids,
                         attention_mask=enc.attention_mask)
        # 2. 解析实体
        tags = out['pred_tags'][0]  # e.g. [0,0,1,2,2,0,...]
        # logging.info(tags)

        id2tag = {v:k for k,v in {'B-crime':0,'I-crime':1,
                                  'B-law':2,'I-law':3,
                                  'B-punishment':4,'I-punishment':5,
                                  'O':6}.items()}

        entities = []
        current_ent, current_type = '', None

        logging.info("Token\tTag  --NER_CLS")
        for idx, t in enumerate(tags):
            tag = id2tag[t]
            if idx<len(question): logging.info(f"{question[idx]}\t{tag}")
            if tag.startswith('B-'):
                if current_ent:
                    entities.append((current_type, current_ent))
                current_type = tag[2:]
                current_ent  = question[idx]
            elif tag.startswith('I-') and current_type == tag[2:]:
                current_ent += question[idx]
            else:
                if current_ent:
                    entities.append((current_type, current_ent))
                    current_ent, current_type = '', None
        if current_ent:
            entities.append((current_type, current_ent))

        # 3. 关系分类
        rel_id = out['rel_logits'].argmax(dim=-1).item() + 1  # 恢复 1–7
        return entities, str(rel_id)

# 简单测试
if __name__ == '__main__':
    qc = NER_CLS()
    # print(qc.classify("盗窃罪怎么判刑"))
    print(qc.classify("危害公共安全罪属于什么罪名"))

