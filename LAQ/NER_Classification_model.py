"""
模型定义：多任务联合实体识别与关系分类模型
ELECTRA+BiLSTM+CRF
结构：
    1. 编码层：加载预训练 ELECTRA/BERT 模型，输出序列特征与 [CLS] 向量。
    2. 实体识别头：BiLSTM → Dropout → Linear → CRF，用于字符级 BIO 序列标注。
    3. 关系分类头：Dropout → Linear，用于将 [CLS] 向量映射为 1–7 的关系类别。

输入：
    input_ids, attention_mask — tokenizer 编码得到
    labels_ner（可选） — BIO 标签序列，shape=(B, L)
    labels_rel（可选） — 关系分类标签，shape=(B,)

输出：
    dict {
      loss: 总损失（实体 + 关系），
      rel_logits: 关系分类得分 (B, 7)，
      pred_tags: CRF 解码后的实体预测序列（List[List[int]]）
    }

引用：
    - https://github.com/hertz-pj/BERT-BiLSTM-CRF-NER-pytorch
    - https://github.com/hoanglocla9/bert-jointly-relation-entity-extraction
"""
import torch
import torch.nn as nn
from transformers import ElectraModel, AutoModel
from TorchCRF import CRF  # batch-first 版本
import logging

class JointEntityRelationModel(nn.Module):
    def __init__(self,
                 encoder_name='./chinese-legal-electra-small-discriminator',  # 默认用的预训练模型为 hfl/chinese-legal-electra-small-discriminator（已下载到本地）
                 num_entity_tags=7,
                 num_relations=7,
                 lstm_hidden=256,
                 lstm_layers=1,
                 dropout=0.1):
        super().__init__()

        # 1. 编码器
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.enc_hidden = self.encoder.config.hidden_size

        # 2. BiLSTM + CRF
        self.lstm = nn.LSTM(
            input_size=self.enc_hidden,
            hidden_size=lstm_hidden // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout_ner = nn.Dropout(dropout)
        self.fc_ner = nn.Linear(lstm_hidden, num_entity_tags)
        self.crf = CRF(num_entity_tags)  # 默认 batch_first

        # 3. 关系分类头
        self.dropout_rel = nn.Dropout(dropout)
        self.fc_rel = nn.Linear(self.enc_hidden, num_relations)

    def forward(self,
                input_ids,
                attention_mask,
                labels_ner=None,
                labels_rel=None):

        # 0. Transformer 编码
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state  # (B, L, H)
        cls_out = seq_out[:, 0, :]           # (B, H)

        # 1. 关系分类分支
        rel_logits = self.fc_rel(self.dropout_rel(cls_out))
        loss_rel = None
        if labels_rel is not None:
            loss_rel = nn.CrossEntropyLoss()(rel_logits, labels_rel)

        # 2. 实体识别分支
        lstm_out = self.lstm(seq_out)[0]               # (B, L, lstm_hidden)
        ner_feats = self.fc_ner(self.dropout_ner(lstm_out))  # (B, L, num_entity_tags)

        # batch-first CRF: emissions (B, L, C), mask (B, L)
        mask = attention_mask.bool()
        loss_ner = None
        if labels_ner is not None:
            # CRF 前向计算 loss
            loss_ner = -self.crf(ner_feats, labels_ner, mask=mask).mean()

        # CRF 解码，返回 List[List[int]]，长度=batch_size，每个子列表长度=valid_len
        pred_tags = self.crf.viterbi_decode(ner_feats, mask)

        # 合并损失
        loss = None
        if loss_rel is not None and loss_ner is not None:
            loss = loss_rel + loss_ner
            logging.info(f'loss_rel: {loss_rel}\nloss_ner: {loss_ner}\ntotal_loss: {loss}  --NER_CLS_model')

        return {
            'loss': loss,
            'rel_logits': rel_logits,
            'pred_tags': pred_tags
        }
