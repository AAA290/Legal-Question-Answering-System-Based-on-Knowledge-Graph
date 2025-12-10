"""
问题类型的序号和分类如下：
| 序号 | 用户输入问句 | 实体识别后问句 |
|--|--|--|
|1| 故意杀人罪的构成要件是什么？ |  [crime] 构成要件 |
|2|诈骗罪可能面临哪些刑罚？有什么量刑标准？| [crime]的[Punishment]
|3|刑法第 232 条规定了什么内容？据民法典第 1165 条应如何处理？|[Law] 内容
|4|盗窃罪涉及哪些法律条文？| [crime]相关的[Law]
|5|刑法第 264 条对应哪些罪名？ |[Law]对应的[crime]
|6|什么是非法拘禁罪？ |[crime]定义
|7|危害国家安全罪包含哪些罪名|[crime]包含的[crime]
"""

import os
import re
import json
import ahocorasick
from openai import OpenAI
from typing import List, Tuple,Optional
import logging
from NER_Classification_use import NER_CLS
from use_crime_prediction import crimes_prediction

class QuestionClassifier:
    def __init__(self):
        # 获取当前文件所在目录
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # 罪名词库路径
        self.crime_path = os.path.join(cur_dir, 'dict/crimes.txt')
        # 读取罪名词库，剔除空行
        with open(self.crime_path, encoding='utf-8') as f:
            self.crime_wds = [line.strip() for line in f if line.strip()]
        self.crime_set = set(self.crime_wds)
        # 利用 Aho-Corasick 构建词典树，提高匹配效率
        self.crime_tree = self.build_actree(list(self.crime_set))

        # 定义法律匹配的正则表达式，匹配《XX法》、以“法”结尾的词和各种条款格式
        self.LAW_PATTERN = re.compile(
            r'(?:'
            r'《([^》]+)法》|'  # 匹配《XX法》
            r'\b\w+法\b|'      # 匹配以“法”结尾的法律名称
            r'(?:第)[零一二三四五六七八九十百千]+条|'  # 匹配中文数字条款
            r'(?:第)\d+条'     # 匹配阿拉伯数字条款
            r')'
        )
        # 定义处罚匹配的正则表达式
        self.PUNISH_PATTERN = re.compile(
            r'(?:'
            r'有期徒刑(?:\d+年|终身)|'  # 匹配“有期徒刑XX年”或“有期徒刑终身”
            r'死刑|'                     # 匹配“死刑”
            r'无期徒刑|'                 # 匹配“无期徒刑”
            r'罚金(?:\d+元)?|'           # 匹配“罚金”或“罚金XXX元”
            r'剥夺政治权利(?:终身|(?:\d+年))|'  # 匹配“剥夺政治权利终身”或“剥夺政治权利X年”
            r'没收(?:个人财产|非法所得)|'  # 匹配“没收个人财产”或“没收非法所得”
            r'拘留(?:\d+天)|'            # 匹配“拘留X天”
            r'行政拘留(?:\d+天)|'        # 匹配“行政拘留X天”
            r'拘役(?:\d+个月)|'          # 匹配“拘役X个月”
            r'缓刑(?:\d+年)|'            # 匹配“缓刑X年”
            r'社区矫正|'                 # 匹配“社区矫正”
            r'警告|'                     # 匹配“警告”
            r'罚款(?:\d+元)?|'           # 匹配“罚款”或“罚款XXX元”
            r'吊销(?:驾驶证|营业执照)|'  # 匹配“吊销驾驶证”或“吊销营业执照”
            r'驱逐出境'                  # 匹配“驱逐出境”
            r')'
        )

        # 定义各种问题类型的疑问词列表
        self.constitution_qwds = ['构成要件', '成立条件', '要素', '满足什么条件', '满足哪些条件', '需要什么条件', '需要具备哪些条件', '如何构成']
        self.punishment_qwds = ['量刑标准', '判刑', '判多少年', '量刑', '刑罚', '处罚', '可能被判', '定罪', '刑罚', '处罚', '后果', '惩罚', '面临', '面临哪些刑罚', '承担', '承担什么责任']
        self.law_content_qwds = ['规定了什么', '包含哪些内容', '具体内容', '是什么意思', '解释', '有什么要求', '如何适用', '规定', '定义']
        self.crime_law_qwds = ['涉及哪些法律', '相关法律条文', '哪些法律条款', '法律依据', '根据什么法律', '适用哪些法律']
        self.law_crime_qwds = ['对应哪些罪名', '属于什么犯罪', '会构成什么罪', '涉及什么罪']
        self.cause_crime_qwds = ['可能涉及哪些罪名', '涉及什么罪', '符合哪个犯罪', '是否构成犯罪', '是否违法']
        self.definition_qwds = ['什么是', '定义', '指什么', '怎么理解', '如何界定', '解释', '含义', '意义', '意思']


    def build_actree(self, wordlist):
        """
        构建 Aho-Corasick 词典树，用于高效匹配罪名关键词
        """
        actree = ahocorasick.Automaton()
        for idx, word in enumerate(wordlist):
            actree.add_word(word, (idx, word))
        actree.make_automaton()
        return actree

    def extract_crime_entities(self, question):
        """使用AC自动机提取罪名"""
        crimes = []
        for end_index, (_, crime) in self.crime_tree.iter(question):
            start_pos = end_index - len(crime) + 1
            if question[start_pos:end_index+1] == crime:
                crimes.append(crime)
        return list(set(crimes))  # 去重

    def extract_entities(self, question):
        """
        实体提取：
         1. 利用 Aho-Corasick 词典树识别罪名（实体类型为 'crime'）
         2. 利用正则表达式匹配法律相关实体（实体类型为 'law'）
        """
        entities = []
        # 匹配罪名
        crimes = self.extract_crime_entities(question)
        entities.extend([('crime', crime) for crime in crimes])

        # 利用正则表达式匹配法律实体
        for m in re.finditer(self.LAW_PATTERN, question):
            match_str = m.group(0)
            entities.append(('law', match_str))

        for m in re.finditer(self.PUNISH_PATTERN, question):
            match_str = m.group(0)
            entities.append(('punishment', match_str))

        return entities

    def determine_question_type(self, question, entities):
        """
        根据问题中的疑问词和已识别的实体判断问题类型
        """
        pattern = ''

        # 如果包含定义类疑问词
        if any(qw in question for qw in self.definition_qwds):
            if any(ent[0] == 'crime' for ent in entities):
                # [crime]定义
                pattern = '6'
            elif any(ent[0] == 'law' for ent in entities):
                # [law]内容
                pattern = '3'
        # 如果包含量刑相关疑问词
        elif any(qw in question for qw in self.punishment_qwds):
            if any(ent[0] == 'crime' for ent in entities):
                # [crime]的[punishment]
                pattern = '2'
        # 如果包含构成要件相关疑问词
        elif any(qw in question for qw in self.constitution_qwds):
            if any(ent[0] == 'crime' for ent in entities):
                # [crime]构成要件
                pattern = '1'
        # 如果包含法律内容疑问词
        elif any(qw in question for qw in self.law_content_qwds):
            if any(ent[0] == 'law' for ent in entities):
                # [law]内容
                pattern = '3'
#------------------------以下都是还需要优化的，不完善！！！！！！！！！！！！！！！！！！！！！------------------#
        # 如果包含涉及法律疑问词
        elif any(qw in question for qw in self.crime_law_qwds):
            # [crime]涉及哪些[law]
            pattern = '4'
        # 如果包含对应罪名疑问词
        elif any(qw in question for qw in self.law_crime_qwds):
            # [law]对应哪些[crime]
            pattern = '5'
        # 如果包含可能涉及罪名的疑问词
        elif any(qw in question for qw in self.cause_crime_qwds):
            # [crime]包含的[crime]
            pattern = '7'
#------------------------记得优化！！！！！------------------------------------------------------------#
        return pattern

    def call_chatgpt_api(self, question: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
        """
        调用ChatGPT API进行实体识别和问题分类
        返回格式: (entities, pattern)
        """
        # 构造系统提示词（指导模型如何分析问题）
        system_prompt = """你是一个专业的法律问答系统分析员，请严格按以下要求处理用户问题：
            1. 识别问题中的法律实体：包括罪名(如"盗窃罪")、法律条文(如"刑法第232条")、处罚类型(如“有期徒刑三年”)等
            识别为罪名一定要是正式的罪名，如果文本中存在如“偷盗”等非正式表达，请先预测是什么正式的罪名并识别为该正式罪名如“盗窃”
            正式罪名如下（有且仅有这些）：
            串通投标
            交通肇事
            介绍贿赂
            以危险方法危害公共安全
            传授犯罪方法
            传播性病
            传播淫秽物品
            伪证
            伪造、倒卖伪造的有价票证
            伪造、变造、买卖国家机关公文、证件、印章
            伪造、变造、买卖武装部队公文、证件、印章
            伪造、变造居民身份证
            伪造、变造金融票证
            伪造公司、企业、事业单位、人民团体印章
            伪造货币
            侮辱
            侵占
            侵犯著作权
            保险诈骗
            信用卡诈骗
            倒卖文物
            倒卖车票、船票
            假冒注册商标
            冒充军人招摇撞骗
            出售、购买、运输假币
            利用影响力受贿
            制作、复制、出版、贩卖、传播淫秽物品牟利
            制造、贩卖、传播淫秽物品
            动植物检疫徇私舞弊
            劫持船只、汽车
            包庇毒品犯罪分子
            协助组织卖淫
            单位受贿
            单位行贿
            危险物品肇事
            危险驾驶
            受贿
            合同诈骗
            失火
            妨害作证
            妨害信用卡管理
            妨害公务
            容留他人吸毒
            对单位行贿
            对非国家工作人员行贿
            寻衅滋事
            巨额财产来源不明
            帮助毁灭、伪造证据
            帮助犯罪分子逃避处罚
            开设赌场
            引诱、容留、介绍卖淫
            引诱、教唆、欺骗他人吸毒
            强制猥亵、侮辱妇女
            强奸
            强迫交易
            强迫他人吸毒
            强迫劳动
            强迫卖淫
            徇私枉法
            徇私舞弊不征、少征税款
            徇私舞弊不移交刑事案件
            打击报复证人
            扰乱无线电通讯管理秩序
            投放危险物质
            抢劫
            抢夺
            拐卖妇女、儿童
            拐骗儿童
            拒不执行判决、裁定
            拒不支付劳动报酬
            招摇撞骗
            招收公务员、学生徇私舞弊
            持有、使用假币
            持有伪造的发票
            挪用公款
            挪用特定款物
            挪用资金
            掩饰、隐瞒犯罪所得、犯罪所得收益
            提供侵入、非法控制计算机信息系统程序、工具
            收买被拐卖的妇女、儿童
            放火
            故意伤害
            故意杀人
            故意毁坏财物
            敲诈勒索
            污染环境
            洗钱
            滥伐林木
            滥用职权
            爆炸
            猥亵儿童
            玩忽职守
            生产、销售不符合安全标准的食品
            生产、销售伪劣产品
            生产、销售伪劣农药、兽药、化肥、种子
            生产、销售假药
            生产、销售有毒、有害食品
            盗伐林木
            盗掘古文化遗址、古墓葬
            盗窃
            盗窃、侮辱尸体
            盗窃、抢夺枪支、弹药、爆炸物
            盗窃、抢夺枪支、弹药、爆炸物、危险物质
            破坏交通工具
            破坏交通设施
            破坏广播电视设施、公用电信设施
            破坏易燃易爆设备
            破坏生产经营
            破坏电力设备
            破坏监管秩序
            破坏计算机信息系统
            票据诈骗
            私分国有资产
            窃取、收买、非法提供信用卡信息
            窝藏、包庇
            窝藏、转移、收购、销售赃物
            窝藏、转移、隐瞒毒品、毒赃
            组织、强迫、引诱、容留、介绍卖淫
            组织、领导、参加黑社会性质组织
            组织、领导传销活动
            组织卖淫
            经济犯
            绑架
            编造、故意传播虚假恐怖信息
            职务侵占
            聚众冲击国家机关
            聚众哄抢
            聚众扰乱公共场所秩序、交通秩序
            聚众扰乱社会秩序
            聚众斗殴
            脱逃
            虐待
            虐待被监管人
            虚开发票
            虚开增值税专用发票、用于骗取出口退税、抵扣税款发票
            虚报注册资本
            行贿
            诈骗
            诬告陷害
            诽谤
            贪污
            贷款诈骗
            赌博
            走私
            走私、贩卖、运输、制造毒品
            走私国家禁止进出口的货物、物品
            走私废物
            走私普通货物、物品
            走私武器、弹药
            走私珍贵动物、珍贵动物制品
            过失以危险方法危害公共安全
            过失投放危险物质
            过失损坏广播电视设施、公用电信设施
            过失损坏武器装备、军事设施、军事通信
            过失致人死亡
            过失致人重伤
            违法发放贷款
            逃税
            遗弃
            重大劳动安全事故
            重大责任事故
            重婚
            金融凭证诈骗
            销售假冒注册商标的商品
            隐匿、故意销毁会计凭证、会计帐簿、财务会计报告
            集资诈骗
            非国家工作人员受贿
            非法买卖、运输、携带、持有毒品原植物种子、幼苗
            非法买卖制毒物品
            非法侵入住宅
            非法出售发票
            非法制造、买卖、运输、储存危险物质
            非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物
            非法制造、出售非法制造的发票
            非法制造、销售非法制造的注册商标标识
            非法占用农用地
            非法吸收公众存款
            非法处置查封、扣押、冻结的财产
            非法拘禁
            非法持有、私藏枪支、弹药
            非法持有毒品
            非法捕捞水产品
            非法携带枪支、弹药、管制刀具、危险物品危及公共安全
            非法收购、运输、出售珍贵、濒危野生动物、珍贵、濒危野生动物制品
            非法收购、运输、加工、出售国家重点保护植物、国家重点保护植物制品
            非法收购、运输盗伐、滥伐的林木
            非法狩猎
            非法猎捕、杀害珍贵、濒危野生动物
            非法生产、买卖警用装备
            非法生产、销售间谍专用器材
            非法种植毒品原植物
            非法组织卖血
            非法经营
            非法获取公民个人信息
            非法获取国家秘密
            非法行医
            非法转让、倒卖土地使用权
            非法进行节育手术
            非法采伐、毁坏国家重点保护植物
            非法采矿
            骗取贷款、票据承兑、金融票证
            高利转贷
            2. 判断问题类型并生成对应模板,模板仅有如下几类，一定要分类为以下类型的一种(返回1-6序号中的一个)：
            | 序号 | 用户输入问句 | 实体识别后问句 |
            |--|--|--|
            |1| 故意杀人罪的构成要件是什么？ |  [crime] 构成要件 |
            |2|诈骗罪可能面临哪些刑罚？有什么量刑标准？| [crime]的[Punishment]  
            |3|刑法第 232 条规定了什么内容？据民法典第 1165 条应如何处理？|[Law] 内容  
            |4|盗窃罪涉及哪些法律条文？| [crime]相关的[Law]  
            |5|刑法第 264 条对应哪些罪名？ |[Law]对应的[crime]  
            |6|什么是非法拘禁罪？ |[crime]定义  
            3. 按指定JSON格式返回结果，包含entities和pattern字段
    
            示例输入1：
            偷东西有哪条法规规定了量刑标准啊
            示例输出1：
            {
                "entities": [("crime", "盗窃"),("law","法规"),("punishment","量刑标准")],
                "pattern": "4"
            }
            示例输入2：
            有人故意殴打他人致使重伤算是故意伤害了，会怎么样判刑
            示例输出2：
            {
                "entities": [("crime", "故意伤害"),("punishment","判刑")],
                "pattern": "2"
            }
            """

        try:
            os.environ["http_proxy"] = "http://localhost:7890"
            os.environ["https_proxy"] = "http://localhost:7890"
            client = OpenAI(
                api_key = os.environ.get("chatanywhere_API_KEY"),
                base_url="https://api.chatanywhere.tech/v1"
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"}
            )

            # 解析API响应
            result = json.loads(response.choices[0].message.content)
            # 正确提取entities的代码
            entities = []
            if isinstance(result.get("entities"), list):  # 确保entities是列表
                for entity_pair in result["entities"]:
                    if isinstance(entity_pair, list) and len(entity_pair) == 2:  # 确保每个实体是[key, value]格式
                        entity_type, entity_value = str(entity_pair[0]), str(entity_pair[1])
                        entities.append((entity_type, entity_value))
            # print("提取后的entities:", entities)

            pattern = result.get("pattern", None)

            return entities, pattern

        except Exception as e:
            logging.error(f"调用ChatGPT API失败: {str(e)}")
            return [], None


    def classify(self, question):
        """
        主分类函数：
         1. 首先使用规则匹配（罪名词库和正则）提取实体；
         2. 根据疑问词及实体信息确定问题类型（构造模式字符串）；
         3. 如果规则匹配结果不理想（实体为空或问题类型未知），则调用训练的ELECTRA_BiLSTM_CRF模型进行补充识别。
        """
        # 1. 使用规则匹配提取实体
        entities = self.extract_entities(question)
        # 2. 根据问题关键词和已提取实体判断问题类型
        pattern = self.determine_question_type(question, entities)

        if not entities or not pattern:
            # 使用ELECTRA_BiLSTM_CRF多任务模型
            print('使用ELECTRA_BiLSTM_CRF模型\n')
            qc = NER_CLS()
            entities, pattern = qc.classify(question)
            if not entities:
            # 如果ELECTRA_BiLSTM_CRF还是识别不出来，就调用gpt-3.5-turbo进行补充识别
                entities, pattern = self.call_chatgpt_api(question)
            if entities:
                if entities[0][0]=='crime' and entities[0][1] not in self.crime_set:
                # 如果是crime且不在罪名库则进行罪名预测
                    print('预测')
                    result = crimes_prediction(entities[0][1])
                    entities = [('crime',result[0][0]+'罪')]

        return entities, pattern

if __name__ == "__main__":
    qc = QuestionClassifier()
    # 测试示例
    test_questions = [
        "盗窃罪是什么意思",
        "盗窃罪怎么判刑",
        "偷东西的量刑标准"
    ]
    for q in test_questions:
        result = qc.classify(q)
        print("输入问题：", q)
        print("返回结果：", result)

