"""
使用 微调后的模型 用于预测罪名 的接口函数
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_SAVE_PATH = "./model/multi_label_crime_model"  # kaggle训练好的模型 的存放地址
class CrimePredictor:
    def __init__(self, model_path= MODEL_SAVE_PATH ):
        # 加载训练好的组件
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.labels = self._load_labels(model_path)

    def _load_labels(self, model_path):
        # 从文件加载罪名标签
        with open(f'{model_path}/labels.txt', 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]

    def predict(self, text, initial_threshold=0.5, min_threshold=0.2, step=0.05):
        """ 新增动态阈值调整机制 """
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.sigmoid(outputs.logits)[0]
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # 动态阈值算法
        current_threshold = initial_threshold
        while current_threshold >= min_threshold:
            # 获取当前阈值下的预测结果
            valid_indices = (probs > current_threshold).nonzero(as_tuple=True)[0]
            if 1 <= len(valid_indices):  # 理想情况直接返回
                return [(self.labels[i], probs[i].item()) for i in valid_indices]
            # elif len(valid_indices) > 3:  # 取概率最高的前3个
            #     return [(self.labels[sorted_indices[i]], sorted_probs[i].item())
            #             for i in range(3)]
            current_threshold -= step  # 阈值递减

        # 保底策略：返回概率最高的3个（即使低于最低阈值）
        return [(self.labels[sorted_indices[i]], sorted_probs[i].item())
                for i in range(min(3, len(self.labels)))]

def crimes_prediction(text):
    # 初始化预测器
    predictor = CrimePredictor()
    # 执行预测
    results = predictor.predict(text)
    return results

if __name__ == "__main__":
    while True:
        case_text = input("请输入案件事实描述（输入q退出）:\n")
        if case_text.lower() == 'q':
            break
        results = crimes_prediction(case_text)

        print("\n预测罪名及概率：")
        for label, prob in results:
            print(f"- {label}: {prob:.2%}")