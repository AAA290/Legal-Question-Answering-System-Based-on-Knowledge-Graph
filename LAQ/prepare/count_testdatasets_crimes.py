"""
用于统计crimes_facts.json中accusation种类有多少，有哪些
以确定该数据集数据是否足够多样
"""

import json

def count_and_save_crimes(input_file, output_file):
    # 用于存储所有罪名（自动去重）
    crimes = set()

    # 读取JSON文件并统计罪名
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # 提取罪名列表并添加到集合
                for crime in data["meta"]["accusation"]:
                    crimes.add(crime)
            except (KeyError, json.JSONDecodeError) as e:
                print(f"解析行时出错：{line.strip()}，错误：{str(e)}")

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for crime in sorted(crimes):  # 按字母排序后写入
            f.write(crime + '\n')

    # 输出统计结果
    print(f"共发现 {len(crimes)} 种不同罪名")
    print(f"罪名列表已保存至 {output_file}")

# 使用示例
count_and_save_crimes("../data/crimes_facts.json", "../output/crimeprediction_crimes.txt")