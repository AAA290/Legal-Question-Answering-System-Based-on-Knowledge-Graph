"""
数据集一共有217016个数据，其中一共有快200种不同的accusation，
我现在希望用模型对于输入的fact预测accusation，但是该数据集太大了我希望使用 Select_Num = 2500 个数据进行训练，
本来是取前2500个数据，但是该数据集同一accusation比较集中，导致选取前2500数据只有2种accusation

本文件的方法：
通过贪婪集合覆盖算法确保所有约200种指控类型至少在2500个样本中出现一次，随后随机补充剩余样本

  步骤	            描述	                   预期效果
加载与预处理	解析JSON，提取类型与映射	准备数据，获取独特类型集
贪婪集合覆盖	选择最小条目集覆盖所有类型	确保每个类型至少出现一次
随机补充	   从剩余条目中随机选择至2500条	达到样本量，保持一定多样性
输出结果	     提供2500条目索引列表	     用户可据此提取训练数据

覆盖保证：贪婪算法确保每个独特指控类型至少出现一次，满足用户“最好所有种类都有”的需求。但若某些类型条目极少，可能需检查覆盖情况。
多样性与分布：随机补充可能因数据分组导致偏向大组，未来可优化为按组比例采样，但当前方法已基本满足需求。
计算效率：对217,016条数据，贪婪算法每次迭代需扫描所有条目，时间复杂度为O(NMlog M)，其中N为条目数，M为类型数（约200），在现代计算机上应可接受。
"""

import json
import random

# 选择的样本数目
Select_Num = 2000

# 加载JSON文件（假设每行一个独立的JSON对象）
all_entries = []
with open('../data/crimes_facts.json', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            entry = json.loads(line)
            all_entries.append(entry)

# 提取所有独特指控类型和条目到类型映射
unique_types = set()
entry_to_types = {}
for idx, entry in enumerate(all_entries):
    types = set(entry['meta']['accusation'])
    unique_types.update(types)
    entry_to_types[idx] = types

# 贪婪集合覆盖选择最小覆盖集
covered_types = set()
selected_entries = []
while len(covered_types) < len(unique_types):
    max_coverage = 0
    best_entry = -1
    for idx in range(len(all_entries)):
        if idx in selected_entries:
            continue
        coverage = len(entry_to_types[idx] - covered_types)
        if coverage > max_coverage:
            max_coverage = coverage
            best_entry = idx
    if best_entry == -1:
        break  # 无法覆盖更多类型
    selected_entries.append(best_entry)
    covered_types.update(entry_to_types[best_entry])

# 检查是否覆盖所有类型
if len(covered_types) < len(unique_types):
    print("警告：无法覆盖所有指控类型。")
else:
    # 随机补充至5000条
    total_samples = Select_Num
    current_selected = len(selected_entries)
    if current_selected >= total_samples:
        sample_indices = selected_entries[:total_samples]
    else:
        all_indices = set(range(len(all_entries)))
        selected_set = set(selected_entries)
        remaining_indices = all_indices - selected_set
        additional_samples_needed = total_samples - current_selected
        random_selected = random.sample(list(remaining_indices), additional_samples_needed)
        sample_indices = selected_entries + random_selected

    # 提取选中的样本
    selected_samples = [all_entries[idx] for idx in sample_indices]

    # 输出到文件
    with open(f'../data/selected_{Select_Num}_samples.json', 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')  # 每个样本另存为一行

    print(f"已保存 {len(selected_samples)} 条数据到 selected_{Select_Num}_samples.json 文件。")