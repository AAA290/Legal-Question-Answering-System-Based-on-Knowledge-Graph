from py2neo import Graph,Node,Relationship
import json

# 配置Neo4j连接
URL = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

class Legalkg:
    def __init__(self):
        self.data_path = 'data/kg_crime.json'
        self.graph = Graph(URL, auth=(USER,PASSWORD))

    def clear_all(self):
        self.graph.delete_all() # 清空图谱
        print('初始化成功')

    # 读取数据集,创建图谱，导出数据
    def create_graph(self):
        f_crime = open('crime.txt', 'w+')
        # f_law = open('law.txt', 'w+'
        for data in open(self.data_path, encoding='utf-8'):
            data_json = json.loads(data)

            node1 = Node('Crime',name = data_json['crime_small'],link = data_json['crime_link'], concept = data_json['gainian'][0])
            self.graph.merge(node1,'Crime','name')  # merge避免重复
            f_crime.write(data_json['crime_small'])

            node2 = Node('Crime',name = data_json['crime_big'])
            self.graph.merge(node2,'Crime','name')  # merge避免重复
            self.graph.create(Relationship(node1,'Belongs_to',node2))

            #法条
            if data_json.get('fatiao'):
                for law in data_json['fatiao']:
                    if not('刑法条文' in law or '推荐：更多' in law or '推荐阅读：更多' in law or law.startswith('[')):
                        lawnode = Node('Law',content = law)
                        self.graph.merge(lawnode,'Law','content')
                        self.graph.create(Relationship(node1,'Based_on',lawnode))

            # 特征要件
            if data_json.get('tezheng'):
                feature_type = '通用特征'  # 默认类型
                for feature in data_json['tezheng']:
                    # 提取特征类型（如"客体要件"）
                    if '推荐：更多' or '呢？' in feature: continue
                    if (data_json['crime_small'] + "的") in feature or feature.endswith("要件") or feature.endswith("方面"):
                        feature_type = feature[-4:]
                        continue
                    if not (feature_type == '通用特征'):
                        feature_node = Node('Constituent', type = feature_type, content = feature)
                        self.graph.merge(feature_node,'Constituent','content')
                        self.graph.create(Relationship(node1, 'Has_constituent', feature_node))
                        feature_type = '通用特征'  # 恢复为默认类型

            # 处罚
            if data_json.get('chufa'):
                for punishment in data_json['chufa']:
                    if '推荐阅读：更多' in punishment: continue
                    punish_node = Node('Punishment', content=punishment)
                    self.graph.merge(punish_node,'Punishment','content')
                    self.graph.create(Relationship(node1, 'Has_punishment', punish_node))

            # 司法解释
            if data_json.get('jieshi'):
                for interpretation in data_json['jieshi']:
                    if '尚未出台' or '推荐：更多' in interpretation: continue
                    interp_node = Node('Interpretation', content=interpretation)
                    self.graph.merge(interp_node,'Interpretation', 'content')
                    self.graph.create(Relationship(node1, 'Has_interpretation', interp_node))

            # # 认定标准
            # if data_json.get('rending'):
            #     for standard in data_json['rending']:
            #         if '按照上述标准认定本罪后该如何处罚？' in standard: break
            #         std_node = Node('Identification', content=standard)
            #         self.graph.create(std_node)
            #         self.graph.create(Relationship(node1, 'Has_identification', std_node))
        print('创建图谱成功')
        f_crime.close()
        # f_law.close()
        return

if __name__ == "__main__":
    lkg = Legalkg()

    # 清空图谱并创建图谱，运行一次即可
    # lkg.clear_all()
    # lkg.create_graph()