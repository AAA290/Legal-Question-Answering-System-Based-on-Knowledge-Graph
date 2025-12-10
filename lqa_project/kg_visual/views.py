from django.shortcuts import render
from django.http import JsonResponse
from neo4j import GraphDatabase
from django.conf import settings
import logging

logging.basicConfig(filename="kg_test.log",
                    filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%M-%Y %H:%M:%S",
                    level=logging.DEBUG)


# 连接 Neo4j 数据库
driver = GraphDatabase.driver(
    settings.NEO4J_CONFIG['URI'],
    auth=(
        settings.NEO4J_CONFIG['USER'],
        settings.NEO4J_CONFIG['PASSWORD']
    )
)

def graph_view(request):
    """渲染图谱展示页面"""
    query = 'MATCH (n)-[r]-(m) RETURN n,r,m LIMIT 500'
    graph_data = search_graph(query).content.decode('utf-8')
    return render(request, "graph.html",{"graph_data": graph_data})

def query_crime(request):
    """
    根据前端传入的罪名名称查询对应的图谱数据
    """
    crime_name = request.GET.get("crime_name", None)
    if not crime_name:
        return JsonResponse({"error": "crime_name 参数缺失"}, status=400)

    query = 'MATCH (n:Crime)-[r]-(m) WHERE n.name = "{0}" RETURN n, r, m'.format(crime_name)

    return search_graph(query)

def expand_node(request):
    """
    根据节点 id 扩展其直接关系网络
    """
    node_id = request.GET.get("node_id", None)
    if not node_id:
        return JsonResponse({"error": "node_id 参数缺失"}, status=400)

    query = 'MATCH (n) WHERE id(n) = {0} MATCH (n)-[r]-(m) RETURN n, r, m'.format(node_id)

    return search_graph(query)

def search_graph(query):
    """
    执行给定的 Cypher 查询语句，从图数据库中检索节点及其关系，并格式化返回结果。

    参数：
        query (str): Cypher 查询语句，格式应为 'MATCH (n)-[r]-(m) RETURN n, r, m'，
                     其中 'n' 代表起始节点，'r' 代表关系，'m' 代表目标节点。
    注意：
        不可以有变动，如'MATCH (c)-[r]-(m) RETURN c, r, m'是不合适的，会导致node = record["n"]产生keyERROR

    返回：
        JsonResponse: 包含以下数据的 JSON 响应：
            - nodes (list): 提取的节点信息，每个节点包含 id、名称（name、displayName）、类别（category）。
            - links (list): 关系信息，包括起始节点、目标节点及关系类型。
    """
    nodes = {}
    links = []
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            node = record["n"]
            # logging.info('record: '+str(record))
            relationship = record["r"]
            # logging.info('relationship: '+ str(relationship))
            other_node = record["m"]

            if node and node.id not in nodes:
                nodes[node.id] = {
                    "id": node.id,
                    "name": str(node.id),  # 使用 id 作为 name
                    "displayName": node.get("name", other_node.get("content", "")),
                    "category": list(node.labels)[0] if node.labels else "Unknown"
                }
            if other_node and other_node.id not in nodes:
                nodes[other_node.id] = {
                    "id": other_node.id,
                    "name": str(other_node.id),  # 使用 id 作为 name
                    "displayName": other_node.get("name", other_node.get("content", "")),
                    "category": list(other_node.labels)[0] if other_node.labels else "Unknown"
                }

            start_id = relationship.start_node.id
            end_id = relationship.end_node.id
            if start_id not in nodes:
                nodes[start_id] = {
                    "id": start_id,
                    "name": f"Node_{start_id}",
                    "category": "Unknown"
                }
            if end_id not in nodes:
                nodes[end_id] = {
                    "id": end_id,
                    "name": f"Node_{end_id}",
                    "category": "Unknown"
                }
            links.append({
                "source": str(start_id),
                "target": str(end_id),
                "relationship": relationship.type
            })

    data = {"nodes": list(nodes.values()), "links": links}
    return JsonResponse(data)