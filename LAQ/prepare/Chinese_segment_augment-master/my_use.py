import os
import jieba
import json
from model import TrieNode
from utils import get_stopwords, load_dictionary, generate_ngram, save_model, load_model
from config import basedir

def load_data(filename, stopwords):
    """
    :param filename:
    :param stopwords:
    :return: 二维数组,[[句子1分词list], [句子2分词list],...,[句子n分词list]]
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word_list = [x for x in jieba.cut(line.strip(), cut_all=False) if x not in stopwords]
            data.append(word_list)
    return data


def load_data_2_root(data):
    print('------> 插入节点')
    for word_list in data:
        # tmp 表示每一行自由组合后的结果（n gram）
        # tmp: [['它'], ['是'], ['小'], ['狗'], ['它', '是'], ['是', '小'], ['小', '狗'], ['它', '是', '小'], ['是', '小', '狗']]
        ngrams = generate_ngram(word_list, 2)
        for d in ngrams:
            root.add(d)
    print('------> 插入成功')


if __name__ == "__main__":
    # 自己修改的部分，将案由数据写入demo.txt
    # with open('data/selected_2000_samples.json', 'r', encoding='utf-8') as f:
    #     data = [json.loads(line) for line in f]
    # with open('data/demo.txt', 'w', encoding='utf-8') as f:
    #     for item in data:
    #         f.write(item['fact'] + '\n')

    root_name = basedir + "/data/root.pkl"
    stopwords = get_stopwords()
    if os.path.exists(root_name):
        root = load_model(root_name)
    else:
        dict_name = basedir + '/data/dict.txt'
        word_freq = load_dictionary(dict_name)
        root = TrieNode('*', word_freq)
        save_model(root, root_name)

    # 加载新的文章
    filename = 'data/demo.txt'
    data = load_data(filename, stopwords)
    # 将新的文章插入到Root中
    load_data_2_root(data)

    # 定义取TOPN个
    topN = 10
    result, add_word = root.find_word(topN)
    # 如果想要调试和选择其他的阈值，可以print result来调整
    # print("\n----\n", result)

    #新增代码，保存result到文件中
    Result_File = "output/results.txt"
    with open(Result_File, 'w', encoding='utf-8') as f:
        for item in result:
            word, score = item
            # 去掉词汇中的下划线 '_'
            word = word.replace('_', '')
            f.write(f"{word}\t{score}\n")

    print(f"结果已成功写入 {Result_File}")

    print("\n----\n", '增加了 %d 个新词, 词语和得分分别为: \n' % len(add_word))
    print('#############################')
    for word, score in add_word.items():
        print(word + ' ---->  ', score)
    print('#############################')

    # 前后效果对比
    test_sentence = '攸县人民检察院指控，2017年3月2日19时许，被告人吴某某到攸县江桥街道“窝里人”饭店找易某谈论事情时，双方发生口角，吴某某顺手拿起桌上的瓷碗向易某砸去，致使易某的脸部、耳朵等部位被划伤。后经鉴定，易某的伤情构成轻伤一级。公诉机关并提供了被害人的陈述、被告人的供述、证人的证言、伤情鉴定意见书等证据，以被告人吴某某犯××向本院提起公诉，请求依法判处。'
    print('添加前：')
    print("".join([(x + '/ ') for x in jieba.cut(test_sentence, cut_all=False) if x not in stopwords]))

    for word in add_word.keys():
        jieba.add_word(word)
    print("添加后：")
    print("".join([(x + '/ ') for x in jieba.cut(test_sentence, cut_all=False) if x not in stopwords]))
