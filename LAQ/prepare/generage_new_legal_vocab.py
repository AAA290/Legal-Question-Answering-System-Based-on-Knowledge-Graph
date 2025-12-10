"""
问题：
    jieba分词太细，导致一些法律领域不应该拆分的被分了，比如有的罪名很长“伪造、变造金融票证罪”，就会被拆分到认不出罪名全称
解决方法1：
    现有的../data/legal_vocab.txt 法律领域词表（来源于：https://github.com/pengxiao-song/LaWGPT/blob/main/resources/legal_vocab.txt）
    不包含这些长罪名，所以在该词表末尾加上罪名库crimes.txt中的所有罪名，存入新的词表new_legal_vocab.txt中
"""
# 读取旧内容
with open('../data/legal_vocab.txt', 'r', encoding='utf-8') as file_a:
    content_a = file_a.read()

# 读取crimes.txt的内容并处理
content_b_processed = []
with open('../dict/crimes.txt', 'r', encoding='utf-8') as file_b:
    for line in file_b:
        # 去除每行末尾的换行符，并在末尾添加“罪”
        processed_line = line.strip() + '罪\n'
        content_b_processed.append(processed_line)

# 将处理后的内容写入new_legal_vocab.txt
with open('../dict/new_legal_vocab.txt', 'w', encoding='utf-8') as file_c:
    # 写入A.txt的内容
    file_c.write(content_a)
    # 写入处理后的B.txt内容
    file_c.writelines(content_b_processed)

print("合并完成，结果已保存到new_legal_vocab.txt")