"""
训练脚本：使用预处理好的 JSONL 数据，训练 JointEntityRelationModel

功能：
    1. 加载 tokenizer 与 BIO、关系标签映射。
    2. 构建 Dataset & DataLoader，支持批量化训练与验证。
    3. 定义优化器（AdamW）与学习率调度（线性衰减）。
    4. 训练循环：前向→计算实体 CRF 损失与关系分类交叉熵→反向→梯度更新。
    5. 每轮打印训练损失，可插入验证评估（实体 F1、关系准确率）。
    6. 训练结束后保存模型权重与 tokenizer。

注意：
    - 实体标签必须严格在 [0, num_tags - 1] 范围内
    - CRF 不支持 batch_first，emissions 需 transpose

输入：
    data/train.jsonl, data/val.jsonl

输出：
    models/joint_model.bin
    models/tokenizer 配置文件
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
from NER_Classification_model import JointEntityRelationModel
from sklearn.model_selection import train_test_split
import logging
import os

os.makedirs('./output', exist_ok=True)
logging.basicConfig(filename='./output/NER_CLS_train_log.txt',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                    level=logging.DEBUG)

def split_dataset(input_path, train_path, val_path,
                  test_size=0.1, random_state=42):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    train, val = train_test_split(data, test_size=test_size, random_state=random_state)
    with open(train_path, 'w', encoding='utf-8') as fw:
        for ex in train:
            fw.write(json.dumps(ex, ensure_ascii=False) + '\n')
    with open(val_path, 'w', encoding='utf-8') as fw:
        for ex in val:
            fw.write(json.dumps(ex, ensure_ascii=False) + '\n')

class NERRelationDataset(Dataset):
    def __init__(self, path, tokenizer, tag2id, max_len=40):  # 最长的问题也就差不多35个字，所以max_len=40足够了
        self.samples = []
        num_tags = len(tag2id)

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                ex = json.loads(line)   # 一行数据
                tokens, ner_tags, rel = ex['tokens'], ex['ner_tags'], ex['relation']

                # 截断
                if len(tokens) > max_len:
                    tokens = tokens[:max_len]
                    ner_tags = ner_tags[:max_len]

                # BIO标签映射
                try:
                    tag_ids = [tag2id[t] for t in ner_tags]  #将BIO标签转化为数字
                except KeyError as e:
                    print(f"非法标签 {e}，已跳过该样本")
                    continue  # 跳过含未知标签的样本

                # 长度 & 范围检查
                if len(tag_ids) > max_len or max(tag_ids) >= num_tags or min(tag_ids) < 0:
                    print(f"[跳过样本] 标签长度或值非法：{tag_ids}")
                    continue

                tag_ids += [tag2id['O']] * (max_len - len(tag_ids))  # 末尾补padding

                enc = tokenizer(tokens,
                                is_split_into_words=True,  # 原数据已分词
                                padding='max_length',
                                truncation=True,
                                max_length=max_len,
                                return_tensors='pt')

                self.samples.append({
                    'input_ids': enc.input_ids.squeeze(0),
                    'attention_mask': enc.attention_mask.squeeze(0),
                    'labels_ner': torch.tensor(tag_ids, dtype=torch.long),
                    'labels_rel': torch.tensor(rel - 1, dtype=torch.long)  # rel-1是因为原来的rel范围是1-7，这里要用0-6
                })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def train(encoder_name,save_path):
    # encoder_name = './chinese-legal-electra-small-discriminator'
    # —— 1. 加载配置
    config = AutoConfig.from_pretrained(encoder_name)

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    # BIO 标签
    tags = ['B-crime','I-crime','B-law','I-law','B-punishment','I-punishment','O']
    tag2id = {t:i for i,t in enumerate(tags)}  # BIO标签和数字的映射字典
    logging.info(f'tag2id: {tag2id}')

    train_ds = NERRelationDataset('data/train.jsonl', tokenizer, tag2id)
    val_ds   = NERRelationDataset('data/val.jsonl', tokenizer, tag2id)
    logging.info(f'train_ds[0]: {train_ds[0]}')

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16)

    model = JointEntityRelationModel(
        encoder_name=encoder_name,
        num_entity_tags=len(tags),
        num_relations=7
    )

    model.config = config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * 10
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    for epoch in range(10):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in train_loader:
            # 1. 数据迁移到设备（GPU/CPU）
            for k,v in batch.items(): batch[k] = v.to(device)
            # 2. 模型前向传播
            out = model(**batch)  # 将批次数据解包为模型输入（input_ids, attention_mask, labels_ner, labels_rel）
            # 3. 获取损失
            loss = out['loss']
            # 4. 梯度清零
            optimizer.zero_grad()  # 清空上一批次的梯度（避免梯度累积）
            # 5. 反向传播
            loss.backward()  # 计算损失对模型参数的梯度
            # 6. 参数更新
            optimizer.step()  # 根据梯度更新模型参数（AdamW优化器）
            # 7. 学习率调整
            scheduler.step()  # 更新学习率（线性衰减调度器，按批次逐步调整）
            # 8. 记录损失
            train_loss += loss.item()  # 累加本批次损失（.item()将Tensor转为Python数值）
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():  # 禁用梯度计算(节省显存、加速验证阶段)
            for batch in val_loader:
                for k,v in batch.items(): batch[k] = v.to(device)
                out = model(**batch)
                val_loss += out['loss'].item()
        print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, "
              f"val_loss={val_loss/len(val_loader):.4f}")

    # 目前是保存最后一个epoch的模型（之后可以改进为保存性能最好的模型）
    model.config.save_pretrained(save_path) # 保存config.json
    torch.save(model.state_dict(), f'{save_path}/pytorch_model.bin') #这样没存config.json
    tokenizer.save_pretrained(save_path)

if __name__ == '__main__':
    # 跑一次把数据集分好就可以了
    # split_dataset('data/questions_preprocessed.jsonl',
    #               'data/train.jsonl',
    #               'data/val.jsonl')
    train(encoder_name='ckiplab/bert-tiny-chinese',save_path = 'model/bert_joint_model/')