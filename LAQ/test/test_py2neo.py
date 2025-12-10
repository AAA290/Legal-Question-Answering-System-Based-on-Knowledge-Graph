from py2neo import Graph
import logging

logging.basicConfig(filename="../output/test_py2neo.log",  # 将日志保存到filename文件中
                    filemode='w',  # 写入的模式
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    level=logging.DEBUG)  # DEBUG及以上的日志信息都会显示

#连接图数据库
USER = "neo4j"
PASSWORD = "password"

graph = Graph("http://localhost:7474",auth=(USER,PASSWORD))

cypher = "match (c:Crime) WHERE c.name = '盗窃罪' RETURN c.name"

print(graph.run(cypher).data())
