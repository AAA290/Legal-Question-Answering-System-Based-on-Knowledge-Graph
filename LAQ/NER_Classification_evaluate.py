"""
模型评估文件
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from seqeval.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from NER_Classification_model import JointEntityRelationModel
from NER_Classification_train import NERRelationDataset

def Evaluate(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_dir = 'model/electra_joint_model'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = JointEntityRelationModel(
        encoder_name=model_dir,
        num_entity_tags=7,
        num_relations=7
    )
    model.load_state_dict(torch.load(f'{model_dir}/pytorch_model.bin',
                                     map_location=device), strict=False)
    model.to(device).eval()

    tags = ['B-crime','I-crime','B-law','I-law','B-punishment','I-punishment','O']
    tag2id = {t:i for i,t in enumerate(tags)}  # BIO标签和数字的映射字典

    val_ds = NERRelationDataset('data/val.jsonl', tokenizer, tag2id)
    val_loader = DataLoader(val_ds, batch_size=16)

    all_true_ner, all_pred_ner = [], []
    all_true_rel, all_pred_rel = [], []
    id2tag = {v:k for k,v in tag2id.items()}

    with torch.no_grad():
        for batch in val_loader:
            # 迁移
            for k in batch: batch[k] = batch[k].to(device)
            out = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'])
            # NER
            for tags_seq, mask_seq, true_seq in zip(
                    out['pred_tags'],
                    batch['attention_mask'],
                    batch['labels_ner']
            ):
                valid_len   = mask_seq.sum().item()
                pred_labels = [id2tag[id] for id in tags_seq[:valid_len-2]]
                # print('pred: '+str(pred_labels))
                true_labels = [id2tag[id.item()] for id in true_seq[:valid_len-2]]
                # print('true: '+str(true_labels))
                all_pred_ner.append(pred_labels)
                all_true_ner.append(true_labels)
            # REL
            preds_rel = out['rel_logits'].argmax(dim=-1).cpu().tolist()
            trues_rel = batch['labels_rel'].cpu().tolist()
            all_pred_rel.extend(preds_rel)
            all_true_rel.extend(trues_rel)

    # 打印结果
    ner_p = precision_score(all_true_ner, all_pred_ner)
    ner_r = recall_score   (all_true_ner, all_pred_ner)
    ner_f = f1_score       (all_true_ner, all_pred_ner)
    rel_p, rel_r, rel_f, _ = precision_recall_fscore_support(
        all_true_rel, all_pred_rel, average='macro'
    )

    print("======== 评估结果 ========")
    print(f"实体识别 —— 精确率: {ner_p:.2%}, 召回率: {ner_r:.2%}, F1: {ner_f:.2%}")
    print(f"关系分类 —— 精确率: {rel_p:.2%}, 召回率: {rel_r:.2%}, F1: {rel_f:.2%}")

if __name__ == '__main__':
    # Evaluate(model_dir='model/electra_joint_model')
    """
    electra_joint_model模型
    ======== 评估结果 ========
    实体识别 —— 精确率: 79.57%, 召回率: 86.05%, F1: 82.68%
    关系分类 —— 精确率: 79.54%, 召回率: 77.06%, F1: 77.80%
    """
    Evaluate(model_dir='model/bert_joint_model')
    """
    bert_joint_model模型
    ======== 评估结果 ========
    实体识别 —— 精确率: 65.93%, 召回率: 69.77%, F1: 67.80%
    关系分类 —— 精确率: 80.97%, 召回率: 73.52%, F1: 75.69%
    """