class GraphSearch:
    def Cypher_search(self,res):
        entities,type = res
        cyp = ''
        entity = ''
        if type == '1':  # [crime] 构成要件
            for i in entities:
                if i[0] == 'crime': entity = i[1]
            cyp = "MATCH (c:Crime)-[:Has_constituent]->(f:Constituent) \
                   WHERE c.name = '{0}' \
                   RETURN f.type,f.content LIMIT 100".format(entity)
        if type == '2':  # [crime]的[Punishment]
            for i in entities:
                if i[0] == 'crime':
                    entity = i[1]
                    break
            cyp = "MATCH (c:Crime)-[:Has_punishment]->(p:Punishment) \
                   WHERE c.name = '{0}' \
                   RETURN p.content LIMIT 100".format(entity)
        if type == '3':  # [Law] 内容
            for i in entities:
                if i[0] == 'law': entity = i[1]
            cyp = "MATCH (l:Law)  \
                   WHERE l.content CONTAINS '{0}' \
                   RETURN l.content LIMIT 100".format(entity)
        if type == '4':  # [crime]相关的[Law]
            for i in entities:
                if i[0] == 'crime': entity = i[1]
            cyp = "MATCH (c:Crime)-[:Based_on]->(l:Law) \
                   WHERE c.name = '{0}罪' \
                   RETURN l.content LIMIT 100".format(entity)
        if type == '5':  # [Law]对应的[crime]
            for i in entities:
                if i[0] == 'law': entity = i[1]
            cyp = "MATCH (c:Crime)-[:Based_on]->(l:Law) \
                   WHERE l.content CONTAINS '{0}' \
                   RETURN c.name LIMIT 100".format(entity)
        if type == '6':  # [crime]定义
            for i in entities:
                if i[0] == 'crime': entity = i[1]
            cyp = "MATCH (c:Crime) \
                   WHERE c.name = '{0}' \
                   RETURN c.concept LIMIT 100".format(entity)
        if type == '7':  # [big crime]包含的[samll crime]
            for i in entities:
                if i[0] == 'crime': entity = i[1]
            cyp = "MATCH (sc:Crime)-[:Belongs_to]->(bc:Crime) \
                   WHERE bc.name = '{0}' \
                   RETURN sc.name LIMIT 100".format(entity)

        result = {'type':type,'cyp':cyp,'entity':entity}
        return result