
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
import torch
from torch.utils.data import Dataset
import os

# ==================== 修改路径部分 ====================
# Kaggle输入文件路径（只读）
ELECTRA_MODEL_PATH = "./chinese-legal-electra-small-discriminator"
DATASET_PATH = "./data/crimes_facts.json"

# Kaggle输出路径（可写）
OUTPUT_DIR = "./outputs"  # 训练过程输出
MODEL_SAVE_PATH = "./model/multi_label_crime_model"  # 最终模型保存
INTERRUPTED_SAVE_PATH = "./model/interrupted_model"  # 中断时保存

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(INTERRUPTED_SAVE_PATH, exist_ok=True)

# ==================== 数据预处理 ====================
def load_json_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            labels = item["meta"]["accusation"] if item["meta"]["accusation"] else ["未知"]
            data.append({
                "text": item["fact"],
                "labels": labels  # 保留所有标签
            })
    return pd.DataFrame(data)

# 加载数据集
df = load_json_data(DATASET_PATH)

# 多标签编码
mlb = MultiLabelBinarizer()
label_matrix = mlb.fit_transform(df['labels'])
num_classes = len(mlb.classes_)

# 数据集划分
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# ==================== 模型配置 ====================
MODEL_NAME = ELECTRA_MODEL_PATH
# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_LENGTH = 256
BATCH_SIZE = 16  # 调整为适合GPU的Batch Size

# ==================== 数据集类 ====================
class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

# 创建数据集
train_dataset = MultiLabelDataset(train_df['text'], label_matrix[train_df.index])
val_dataset = MultiLabelDataset(val_df['text'], label_matrix[val_df.index])

# ==================== 新增回调类 ====================
# 确保检查点也保存tokenizer,而不是只保存模型
class TokenizerSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        # 每次自动保存检查点时执行
        output_dir = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}"
        )
        tokenizer.save_pretrained(output_dir)  # 【核心补丁】保存分词器到检查点目录

# ==================== 模型训练配置 ====================
def compute_metrics(pred):
    logits = pred.predictions
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy()
    labels = pred.label_ids

    print("\n分类报告:")
    print(classification_report(labels, preds, target_names=mlb.classes_, zero_division=0))

    return {
        'f1_micro': f1_score(labels, preds, average='micro'),
        'f1_macro': f1_score(labels, preds, average='macro')
    }

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir = OUTPUT_DIR + '/training_outputs',
    num_train_epochs=10,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir= OUTPUT_DIR + '/logs',
    logging_steps=50,
    learning_rate=5e-5,  # 调整为适合GPU的学习率
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    fp16=True,  # 启用混合精度训练以加速并减少显存占用
    gradient_accumulation_steps=1,  # 整为适合GPU的梯度累积步数
    save_total_limit=3,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[TokenizerSaveCallback()]  # 【关键】添加回调,保证终端进程时保存的模型也保存tokenizer文件
)

try:
    trainer.train()
except KeyboardInterrupt:
    print("训练被中断，正在保存当前进度...")
    model.save_pretrained(INTERRUPTED_SAVE_PATH)
    tokenizer.save_pretrained(INTERRUPTED_SAVE_PATH)

# ==================== 模型使用 ====================
def predict_multi_label(text, model, threshold=0.3):   ###0.3有点太高了
    # 编码文本并移动到模型所在设备
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 核心修复：设备对齐(修复问题:PyTorch模型设备不一致问题)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)
    probs = probs.cpu()  # 将结果移回CPU以便处理
    pred_indices = (probs > threshold).nonzero(as_tuple=True)[1].tolist()

    return [(mlb.classes_[i], probs[0][i].item()) for i in pred_indices]

# ==================== 模型保存 ====================
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

# 手动保存为pytorch_model.bin格式
torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'pytorch_model.bin'))

# 定义测试用例，包含案由和正确的罪名
test_cases = [
    {
        "case": "被告先盗窃商店财物，后持刀威胁店员构成抢劫",
        "correct_labels": ["盗窃罪", "抢劫罪"]
    },
    {
        "case": "被告人翻墙进入事主家中盗窃财物，被事主发现后使用随身携带的工具暴力抗拒抓捕，致使事主轻微伤",
        "correct_labels": ["盗窃罪", "抢劫罪", "妨害公务罪"]
    },
    {
        "case": "被告人加入电信诈骗团伙，通过虚假投资平台诱骗多名被害人投资，涉案金额巨大，同时涉及伪造金融凭证",
        "correct_labels": ["诈骗罪", "伪造金融凭证罪"]
    },
    {
        "case": "被告人酒后驾驶机动车在高速公路上逆行，造成多车连环相撞事故，致3人重伤，现场查获其血液酒精含量超标",
        "correct_labels": ["危险驾驶罪", "交通肇事罪"]
    },
    {
        "case": "被告人经营小作坊，在食品生产过程中非法添加有毒有害物质，销售给多个超市，导致多名消费者食物中毒",
        "correct_labels": ["生产、销售有毒、有害食品罪"]
    },
    {
        "case": "被告人酒后无故辱骂殴打路人，引发多人围观，后又纠集同伙与被害人及其朋友发生聚众斗殴，致多人轻伤",
        "correct_labels": ["寻衅滋事罪", "聚众斗殴罪"]
    },
    {
        "case": "被告人以高额回报为诱饵，未经批准向社会不特定人群非法吸收存款，涉及集资参与人上百人，资金无法偿还",
        "correct_labels": ["非法吸收公众存款罪"]
    },
    {
        "case": "被告人身为公司财务人员，利用职务之便侵占公司巨额资金，并在案发前销毁会计凭证等证据材料试图掩盖犯罪事实",
        "correct_labels": ["职务侵占罪", "毁灭证据罪"]
    },
    {
        "case": "被告人通过网络平台传播淫秽视频，注册会员超过万人，以此收取会员费牟利，情节特别严重",
        "correct_labels": ["传播淫秽物品牟利罪"]
    },
    {
        "case": "被告人私藏多把自制火药枪及上百发子弹，在其住所被警方查获，经鉴定枪支具备杀伤力",
        "correct_labels": ["非法持有、私藏枪支、弹药罪"]
    },
    {
        "case": "偷东西",
        "correct_labels": ["盗窃罪"]
    },
    {
        "case": "被告人在法院判决生效后，拒不履行赔偿义务，转移隐匿财产，导致法院判决无法执行",
        "correct_labels": ["拒不执行判决、裁定罪"]
    }
]

# 遍历测试用例
for idx, test_case in enumerate(test_cases, start=1):
    test_text = test_case["case"]
    correct_labels = test_case["correct_labels"]

    # 预测
    predictions = predict_multi_label(test_text, model)

    # 打印测试结果
    print(f"\n===== 测试用例 {idx} =====")
    print(f"案由：{test_text}")
    print("\n预测结果：")
    for label, prob in predictions:
        print(f"{label}: {prob:.2%}")
    print("\n正确罪名：")
    for label in correct_labels:
        print(f"{label}")

# 保存mlb.classes_
with open( MODEL_SAVE_PATH + '/labels.txt', 'w', encoding='utf-8') as f:
    for label in mlb.classes_:
        f.write(label + '\n')