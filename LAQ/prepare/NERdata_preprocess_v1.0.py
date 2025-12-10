"""
技术路线：
1.通过jieba将数据集中的“fact”部分也就是案由部分分词。
2.统计词频，筛选出高频词汇，给高频词配上对应标签，并将其加入标签库。
    标签有：法律，法条，法律组织，法律角色，案由，法律处罚，罪名，人名，司法名词，地名
3.使用标签库（高频词＋对应的标签）作为数据集训练模型，得到最终的法律领域NER模型

数据来源说明：
    ../data/legal_vocab.txt 法律领域词表
        来源于：https://github.com/pengxiao-song/LaWGPT/blob/main/resources/legal_vocab.txt
    ../data/crimes_facts.json 是CAIL2018数据集第一阶段的test数据集

暂时没用到这个文件！！！
"""

import json
import jieba
# import jieba.posseg as pseg
import re
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------- 配置参数 ---------------------
INPUT_JSON = '../data/selected_2000_samples.json'
LEGAL_VOCAB = '../dict/new_legal_vocab.txt'
CRIMES_FILE = '../dict/crimes.txt'
STOPWORDS_FILE = '../dict/legal_stopwords.txt'
OUTPUT_FILE = '../output/count_words_tfidf.txt'

# --------------------- 自定义停用词 ---------------------
DEFAULT_STOPWORDS = {
    '，', '。',  '；', '：', '“', '”', '（', '）', '的', '了', '是', '在', '和',
    '等', '有', '之', '为', '对', '并', '由', '与', '于', '据', '向', '元', '某','某某'
}

# --------------------- 初始化词典 ---------------------
def load_resources():
    jieba.load_userdict(LEGAL_VOCAB)

    with open(CRIMES_FILE, 'r', encoding='utf-8') as f:
        crimes = {line.strip() for line in f}

    with open(LEGAL_VOCAB, 'r', encoding='utf-8') as f:
        legal_vocab = {line.strip() for line in f}

    stopwords = DEFAULT_STOPWORDS.copy()
    try:
        with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
            stopwords.update(line.strip() for line in f)
    except FileNotFoundError:
        print(f"警告：未找到停用词文件 {STOPWORDS_FILE}，使用默认停用词")

    return legal_vocab,crimes,stopwords

# --------------------- 文本清洗 ---------------------
def clean_text(text):
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # text = re.sub(r'[×xX]{2,}', '某', text)
    text = re.sub(r'[【】（）“”‘’]', '', text)
    return text.strip()

# --------------------- 法条正则匹配增强 ---------------------
LAW_PATTERN = re.compile(
    r'(?:'
    r'《([^》]+)法》|'  # 匹配《XX法》
    r'\b\w+法\b|'  # 匹配以“法”结尾的法律名称
    r'(?:第)[零一二三四五六七八九十百千]+条|'  # 匹配中文数字条款
    r'(?:第)\d+条|'  # 匹配阿拉伯数字条款
    r')'
)

# --------------------- 自动标签规则 ---------------------
def get_auto_label(word,crimes,legal_vocab):
    # 优先匹配法条正则
    if LAW_PATTERN.fullmatch(word):
        return '法条'
    if word in crimes or word.endswith('罪'):
        return '罪名'
    if any(kw in word for kw in ['检察院', '法院', '司法局', '仲裁委']):
        return '法律组织'
    if word in ['被告人','被告','被害人','原告', '原告人', '辩护人', '代理人', '公诉人']:
        return '法律角色'
    if word.endswith(('国', '省', '市', '县', '区', '镇', '村', '路', '小学', '银行')):
        return '地名'
    if re.match(r'^(拘留|有期徒刑|没收|处罚金|判刑)', word):
        return '法律处罚'
    if word in legal_vocab or word.endswith('书'):
        return '司法名词'
    return ''

# --------------------- 动态词频调整 ---------------------
def dynamic_adjust_freq(text):
    """通过相邻词共现频率动态调整词频，返回优化后的分词字符串"""
    # 首次分词
    first_cut = jieba.lcut(text, HMM=True)

    # 统计双词共现频率
    cooccur = defaultdict(int)
    for i in range(len(first_cut) - 1):
        pair = (first_cut[i], first_cut[i + 1])
        cooccur[pair] += 1

    # 对高频组合动态加词
    with open('../output/userdict.txt', 'a', encoding='utf-8') as f:
        for pair, count in cooccur.items():
            if count >= 3:  # 出现3次以上的组合视为短语
                phrase = ''.join(pair)
                jieba.add_word(phrase, freq=10000, tag='n')
                f.write(phrase+'\n')

    return jieba.lcut(text, HMM=True)

# --------------------- TF-IDF计算 ---------------------
def calculate_tfidf(documents):
    # 使用已分词的文档构建TF-IDF
    vectorizer = TfidfVectorizer(
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b',
        tokenizer=lambda x: x.split(),  # 直接使用预先分词的文档
        use_idf=True,
        smooth_idf=True
    )

    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names()

    # 计算全局TF-IDF权重
    tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    sorted_indices = np.argsort(-tfidf_scores)

    return [(feature_names[i], tfidf_scores[i])
            for i in sorted_indices]

# 定义法律领域常见短语模式
LEGAL_PHRASE_PATTERNS = [
    r'发生[纠纷|口角|争执|争吵]',    # 匹配"发生纠纷"等
    r'[互相|双方][殴打|撕打]',  # 匹配"互相殴打"等
    r'[婚姻|合同|债务]纠纷',     # 匹配"婚姻纠纷"等
    r'(?:构成)[轻|重]伤[一二三四五六七八九十]级',
    r'价值人民币(?:共计)?\d+元',
    '故意伤害他人身体'
]

def merge_phrase(text):
    """通过正则合并短语（优化实现）"""
    # 预编译正则表达式
    compiled_patterns = [re.compile(pattern) for pattern in LEGAL_PHRASE_PATTERNS]

    for pattern in compiled_patterns:
        # 查找所有匹配项（非重叠）
        matches = list(pattern.finditer(text))
        replacements = []

        # 先记录所有匹配项的位置和内容
        for match in matches:
            start, end = match.span()
            phrase = match.group()
            # 将短语中的空格替换为连接符
            new_phrase = phrase.replace(' ', '_PH_')  # 使用特殊连接符
            replacements.append((start, end, new_phrase))

        # 从后往前替换避免影响索引
        for start, end, new_phrase in reversed(replacements):
            text = text[:start] + new_phrase + text[end:]

    # 添加合并后的短语到分词词典
    for pattern in LEGAL_PHRASE_PATTERNS:
        if '(' in pattern:  # 处理带选项的模式
            base = re.sub(r'\([^)]+\)', '', pattern).replace('|', '')
            jieba.add_word(base, freq=1000, tag='n')
        else:
            jieba.add_word(pattern, freq=1000, tag='n')

    return text


# Version 1
# --------------------- 主处理流程 ---------------------
def main_process():
    legal_vocab,crimes,stopwords = load_resources()

    # 读取并预处理数据
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    processed_docs = []
    valid_words = defaultdict(int)

    for item in data:
        text = clean_text(item['fact'])

        # Step 1: 预合并高频短语
        text = merge_phrase(text)

        # Step 2: 动态调整词频（返回分词列表）
        words = dynamic_adjust_freq(text)

        filtered_words = []
        for word in words:
            word = word.strip()
            if len(word) < 2: continue
            if word in stopwords: continue
            if re.match(r'^\d+年?$', word): continue
            if word.endswith('某'): continue
            filtered_words.append(word)
            valid_words[word] += 1  # 保留原始词频备用

        processed_docs.append(' '.join(filtered_words))

    # 计算TF-IDF并排序
    tfidf_results = calculate_tfidf(processed_docs)

    # 合并统计结果
    word_stats = []
    for word, tfidf in tfidf_results:
        if word in valid_words:
            word_stats.append((
                word,
                valid_words[word],  # 原始词频
                tfidf              # TF-IDF权重值
            ))

    # 按TF-IDF降序取前1500
    word_stats.sort(key=lambda x: -x[2])
    top_words = word_stats[:1500]

    # 输出结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("词汇\t词频\tTF-IDF\t预标标签\n")
        for word, freq, tfidf in top_words:
            label = get_auto_label(word,crimes,legal_vocab)
            f.write(f"{word}\t{freq}\t{tfidf:.4f}\t{label}\n")

    print(f"处理完成！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main_process()

# """
# version 2
#     修改实现了：
#         仅保留名词和动词
#         动词强制标记为"案由"
#         名词根据规则自动标记
#         处理同一词多标签情况（取最高频标签）
#         保持原有TF-IDF计算逻辑
# """
#
# # --------------------- 主处理流程 ---------------------
# def main_process():
#     stopwords = load_resources()
#
#     # 读取并预处理数据
#     with open(INPUT_JSON, 'r', encoding='utf-8') as f:
#         data = [json.loads(line) for line in f]
#
#     processed_docs = []
#     valid_words = defaultdict(int)
#     word_label_counts = defaultdict(lambda: defaultdict(int))  # 新增：记录词标签分布
#
#     for item in data:
#         text = clean_text(item['fact'])
#         words = jieba.posseg.lcut(text)  # 修改：使用带词性的分词
#
#         filtered_words = []
#         for word_pair in words:
#             word = word_pair.word.strip()
#             flag = word_pair.flag
#
#             # 基础过滤
#             if len(word) < 2: continue
#             if word in stopwords: continue
#             if re.match(r'^\d+年?$', word): continue
#
#             # 词性过滤与标签分配
#             label = None
#             if flag.startswith('v'):  # 动词
#                 label = '案由'
#             elif flag.startswith('n'):  # 名词
#                 label = get_auto_label(word)
#                 # if not label: continue  # 过滤无标签名词
#             else:  # 非名词动词
#                 continue
#
#             # 更新统计
#             filtered_words.append(word)
#             valid_words[word] += 1
#             word_label_counts[word][label] += 1  # 记录标签分布
#
#         processed_docs.append(' '.join(filtered_words))
#
#     # 确定每个词的主标签（出现最多的标签）
#     word_labels = {}
#     for word, counts in word_label_counts.items():
#         max_label = max(counts.items(), key=lambda x: x[1])[0]
#         word_labels[word] = max_label
#
#     # 计算TF-IDF并排序（后续流程保持不变）
#     tfidf_results = calculate_tfidf(processed_docs)
#
#     # 合并统计结果
#     word_stats = []
#     for word, tfidf in tfidf_results:
#         if word in valid_words:
#             word_stats.append((
#                 word,
#                 valid_words[word],  # 原始词频
#                 tfidf              # TF-IDF权重值
#             ))
#
#     # 按TF-IDF降序取前1500
#     word_stats.sort(key=lambda x: -x[2])
#     top_words = word_stats[:1500]
#
#     # 输出结果（修改标签获取方式）
#     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#         f.write("词汇\t词频\tTF-IDF\t预标标签\n")
#         for word, freq, tfidf in top_words:
#             label = word_labels.get(word, '')
#             f.write(f"{word}\t{freq}\t{tfidf:.4f}\t{label}\n")
#
#     print(f"处理完成！结果已保存至 {OUTPUT_FILE}")
#
# if __name__ == "__main__":
#     main_process()