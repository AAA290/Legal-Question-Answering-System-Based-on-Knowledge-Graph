"""
简单通过几个测试用例 测试微调后的模型预测罪名的性能
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer

# ==================== 模型加载 ====================
# 加载分词器和模型
MODEL_SAVE_PATH = "../model/multi_label_crime_model"  # 替换为你的模型保存路径
tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)

mlb = MultiLabelBinarizer()
with open('../model/multi_label_crime_model/labels.txt', 'r', encoding='utf-8') as f:
    mlb.classes_ = [line.strip() for line in f.readlines()]

# ==================== 多标签预测函数 ====================
def predict_multi_label(text, model, mlb, threshold=0.1):
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

# ==================== 测试用例 ====================
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
        "case": "被告人在法院判决生效后，拒不履行赔偿义务，转移隐匿财产，导致法院判决无法执行",
        "correct_labels": ["拒不执行判决、裁定罪"]
    }
]

# ==================== 测试模型 ====================
for idx, test_case in enumerate(test_cases, start=1):
    test_text = test_case["case"]
    correct_labels = test_case["correct_labels"]

    # 预测
    predictions = predict_multi_label(test_text, model, mlb)

    # 打印测试结果
    print(f"\n===== 测试用例 {idx} =====")
    print(f"案由：{test_text}")
    print("\n预测结果：")
    for label, prob in predictions:
        print(f"{label}: {prob:.2%}")
    print("\n正确罪名：")
    for label in correct_labels:
        print(f"{label}")