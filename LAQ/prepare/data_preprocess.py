"""
预处理脚本：将原始 JSON 问答数据转换为序列标注与关系分类训练样本。

功能：
    1. 解析原始 data/questions.json，每条记录包含 “instruction” 与 “output”。
    2. 从 “instruction” 中提取关系标签（1–7）和原始问句。
    3. 从 “output” 中正则抽取实体文本及其类型（crime/law/punishment）。
    4. 按字符级构造 tokens 序列及对应的 BIO 形式 ner_tags。
    5. 划分训练集和验证集，并分别保存为 JSONL 文件（data/train.jsonl、data/val.jsonl）。

输入：
    data/questions.json — 原始标注数据（JSON 数组格式）

输出：
    data/train.jsonl — 训练样本，每行一个 JSON，字段：tokens, ner_tags, relation
    data/val.jsonl   — 验证样本，同上
"""
import json
import re

def preprocess(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    examples = []
    for item in raw:
        # 提取关系标签和原始问句
        parts = item['instruction'].split('，', 1)
        if len(parts) < 2:  # 添加错误处理
            print(f"异常instruction格式: {item['instruction']}")
            continue

        rel_label = int(parts[0])
        question = parts[1].strip()

        # 修正正则表达式
        m = re.match(r"\('(.+)',\s*(\w+)\)", item['output'])
        if not m:
            print(f"未匹配的output: {item['output']}")  # 添加调试信息
            continue

        ent_text, ent_type = m.groups()
        tokens = list(question)
        ner_tags = ['O'] * len(tokens)

        # 标注实体位置
        start = question.find(ent_text)
        if start >= 0:
            for i in range(len(ent_text)):
                ner_tags[start + i] = f"B-{ent_type}" if i == 0 else f"I-{ent_type}"
        else:
            print(f"实体未找到: '{ent_text}' in '{question}'")  # 添加警告

        examples.append({
            'tokens': tokens,
            'ner_tags': ner_tags,
            'relation': rel_label
        })

    # 保存完整预处理数据
    with open(output_path, 'w', encoding='utf-8') as fw:
        for ex in examples:
            fw.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"成功处理 {len(examples)} 条数据，已保存至 {output_path}")  # 添加完成提示

if __name__ == '__main__':
    preprocess('../data/questions_dataset.json', '../data/questions_preprocessed.jsonl')
    # output = "('遗失武器装备罪',crime)"
    # pattern = r"\('(.+)',\s*(\w+)\)"
    # m = re.match(pattern, output)
    # print(m.groups() if m else "No match")