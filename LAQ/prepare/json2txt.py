import os
import json

# JSON 文件和 TXT 文件的目录路径
json_file = '../data/kg_crime.json' # JSON 文件路径
out_file = '../data/kg_crime_clean.txt' # 输出的 TXT 文件路径

# 读取 JSON 文件并转换为 TXT
def convert_json_to_txt(json_file, filename):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    with open(out_file,'w', encoding='utf-8') as fs:
        for item in data:
            fs.write(item['crime_small']+'属于'+item['crime_big']+'。')
            fs.write(item['gainian'][0][:150])
            fs.write(''.join(item['tezheng'][:3]))
            fs.write(''.join(item['chufa'][:3]))
            fs.write(''.join(item['fatiao'])+'\n')

if __name__ == '__main__':
    convert_json_to_txt(json_file,out_file)