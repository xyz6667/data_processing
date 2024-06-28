# -*- coding: utf-8 -*-
import re  # 导入正则表达式模块
import sqlparse  # 导入sql解析模块

# 引入骆驼命名法
import inflection

# 引入词性还原
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wnler = WordNetLemmatizer()  # 创建词性还原对象

# 引入词干提取
from nltk.corpus import wordnet

# 定义一些常量，代表不同的token类型
OTHER = 0
FUNCTION = 1
BLANK = 2
KEYWORD = 3
INTERNAL = 4
TABLE = 5
COLUMN = 6
INTEGER = 7
FLOAT = 8
HEX = 9
STRING = 10
WILDCARD = 11
SUBQUERY = 12
DUD = 13

# 将token类型的数字映射到对应的字符串
ttypes = {0: "OTHER", 1: "FUNCTION", 2: "BLANK", 3: "KEYWORD", 4: "INTERNAL", 5: "TABLE", 6: "COLUMN", 7: "INTEGER",
          8: "FLOAT", 9: "HEX", 10: "STRING", 11: "WILDCARD", 12: "SUBQUERY", 13: "DUD", }

# 定义一个扫描器，用来识别和处理特定的正则表达式
scanner = re.Scanner([(r"\[[^\]]*\]", lambda scanner, token: token), (r"\+", lambda scanner, token: "REGPLU"),
                      (r"\*", lambda scanner, token: "REGAST"), (r"%", lambda scanner, token: "REGCOL"),
                      (r"\^", lambda scanner, token: "REGSTA"), (r"\$", lambda scanner, token: "REGEND"),
                      (r"\?", lambda scanner, token: "REGQUE"),
                      (r"[\.~``;_a-zA-Z0-9\s=:\{\}\-\\]+", lambda scanner, token: "REFRE"),
                      (r'.', lambda scanner, token: None), ])

# 子函数1：将字符串按照定义的规则进行分词
def tokenizeRegex(s):
    results = scanner.scan(s)[0]
    return results

# 子函数2：定义一个SQL语言的解析器
class SqlangParser():
    # 静态方法，将输入的SQL语句进行标准化处理
    @staticmethod
    def sanitizeSql(sql):
        s = sql.strip().lower()  # 去除首尾空格并转为小写
        if not s[-1] == ";":
            s += ';'  # 如果SQL语句最后没有分号，添加一个
        s = re.sub(r'\(', r' ( ', s)  # 将左括号前后添加空格
        s = re.sub(r'\)', r' ) ', s)  # 将右括号前后添加空格
        words = ['index', 'table', 'day', 'year', 'user', 'text']
        for word in words:
            s = re.sub(r'([^\w])' + word + '$', r'\1' + word + '1', s)  # 将特定单词的结尾添加数字1
            s = re.sub(r'([^\w])' + word + r'([^\w])', r'\1' + word + '1' + r'\2', s)  # 将特定单词的前后添加数字1
        s = s.replace('#', '')  # 去除'#'
        return s

    # 解析字符串token
    def parseStrings(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):  # 如果token是一个token列表
            for c in tok.tokens:  # 遍历列表中的每个token
                self.parseStrings(c)  # 递归调用自身进行解析
        elif tok.ttype == STRING:  # 如果token是一个字符串
            if self.regex:
                tok.value = ' '.join(tokenizeRegex(tok.value))  # 对字符串进行正则表达式的分词处理
            else:
                tok.value = "CODSTR"  # 否则，将字符串的值设为"CODSTR"

    # 重命名标识符
    def renameIdentifiers(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):  # 如果token是一个token列表
            for c in tok.tokens:  # 遍历列表中的每个token
                self.renameIdentifiers(c)  # 递归调用自身进行重命名
        elif tok.ttype == COLUMN:  # 如果token是一个列名
            if str(tok) not in self.idMap["COLUMN"]:  # 如果列名不在已有的映射中
                colname = "col" + str(self.idCount["COLUMN"])  # 生成新的列名
                self.idMap["COLUMN"][str(tok)] = colname  # 在映射中添加新的列名
                self.idMapInv[colname] = str(tok)  # 在反向映射中添加新的列名
                self.idCount["COLUMN"] += 1  # 列名计数器加一
            tok.value = self.idMap["COLUMN"][str(tok)]  # 将token的值设为新的列名
        elif tok.ttype == TABLE:  # 如果token是一个表名
            if str(tok) not in self.idMap["TABLE"]:  # 如果表名不在已有的映射中
                tabname = "tab" + str(self.idCount["TABLE"])  # 生成新的表名
                self.idMap["TABLE"][str(tok)] = tabname  # 在映射中添加新的表名
                self.idMapInv[tabname] = str(tok)  # 在反向映射中添加新的表名
                self.idCount["TABLE"] += 1  # 表名计数器加一
            tok.value = self.idMap["TABLE"][str(tok)]  # 将token的值设为新的表名
        elif tok.ttype == FLOAT:  # 如果token是一个浮点数
            tok.value = "CODFLO"  # 将token的值设为"CODFLO"
        elif tok.ttype == INTEGER:  # 如果token是一个整数
            tok.value = "CODINT"  # 将token的值设为"CODINT"
        elif tok.ttype == HEX:  # 如果token是一个十六进制数
            tok.value = "CODHEX"  # 将token的值设为"CODHEX"

    # 定义哈希函数，用于生成对象的哈希值
    def __hash__(self):
        return hash(tuple([str(x) for x in self.tokensWithBlanks]))

    # 类的初始化函数
    def __init__(self, sql, regex=False, rename=True):
        # 对输入的SQL语句进行清洗
        self.sql = SqlangParser.sanitizeSql(sql)
        # 初始化一些用于记录标识符映射和计数的字典
        self.idMap = {"COLUMN": {}, "TABLE": {}}
        self.idMapInv = {}
        self.idCount = {"COLUMN": 0, "TABLE": 0}
        self.regex = regex  # 记录是否使用正则表达式
        self.parseTreeSentinel = False  # 初始化解析树哨兵
        self.tableStack = []  # 初始化表栈
        # 对清洗后的SQL语句进行解析
        self.parse = sqlparse.parse(self.sql)
        self.parse = [self.parse[0]]
        # 调用一系列函数对解析树进行处理
        self.removeWhitespaces(self.parse[0])
        self.identifyLiterals(self.parse[0])
        self.parse[0].ptype = SUBQUERY
        self.identifySubQueries(self.parse[0])
        self.identifyFunctions(self.parse[0])
        self.identifyTables(self.parse[0])
        self.parseStrings(self.parse[0])
        if rename:
            self.renameIdentifiers(self.parse[0])
        # 获取处理后的token列表
        self.tokens = SqlangParser.getTokens(self.parse)

    # 静态方法，用于获取解析树中的token
    @staticmethod
    def getTokens(parse):
        flatParse = []
        for expr in parse:
            for token in expr.flatten():
                if token.ttype == STRING:
                    flatParse.extend(str(token).split(' '))
                else:
                    flatParse.append(str(token))
        return flatParse

    # 移除解析树中的空白token
    def removeWhitespaces(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            tmpChildren = []
            for c in tok.tokens:
                if not c.is_whitespace:
                    tmpChildren.append(c)
            tok.tokens = tmpChildren
            for c in tok.tokens:
                self.removeWhitespaces(c)

    # 识别解析树中的子查询
    def identifySubQueries(self, tokenList):
        isSubQuery = False
        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                subQuery = self.identifySubQueries(tok)
                if (subQuery and isinstance(tok, sqlparse.sql.Parenthesis)):
                    tok.ttype = SUBQUERY
            elif str(tok) == "select":
                isSubQuery = True
        return isSubQuery

    # 识别解析树中的字面量
    def identifyLiterals(self, tokenList):
        blankTokens = [sqlparse.tokens.Name, sqlparse.tokens.Name.Placeholder]
        blankTokenTypes = [sqlparse.sql.Identifier]
        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                tok.ptype = INTERNAL
                self.identifyLiterals(tok)
            elif (tok.ttype == sqlparse.tokens.Keyword or str(tok) == "select"):
                tok.ttype = KEYWORD
            elif (tok.ttype == sqlparse.tokens.Number.Integer or tok.ttype == sqlparse.tokens.Literal.Number.Integer):
                tok.ttype = INTEGER
            elif (tok.ttype == sqlparse.tokens.Number.Hexadecimal or tok.ttype == sqlparse.tokens.Literal.Number.Hexadecimal):
                tok.ttype = HEX
            elif (tok.ttype == sqlparse.tokens.Number.Float or tok.ttype == sqlparse.tokens.Literal.Number.Float):
                tok.ttype = FLOAT
            elif (tok.ttype == sqlparse.tokens.String.Symbol or tok.ttype == sqlparse.tokens.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Symbol):
                tok.ttype = STRING
            elif (tok.ttype == sqlparse.tokens.Wildcard):
                tok.ttype = WILDCARD
            elif (tok.ttype in blankTokens or isinstance(tok, blankTokenTypes)):
                tok.ttype = COLUMN

    # 识别解析树中的函数
    def identifyFunctions(self, tokenList):
        for tok in tokenList.tokens:
            if (isinstance(tok, sqlparse.sql.Function)):
                self.parseTreeSentinel = True
            elif (isinstance(tok, sqlparse.sql.Parenthesis)):
                self.parseTreeSentinel = False
            if self.parseTreeSentinel:
                tok.ttype = FUNCTION
            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyFunctions(tok)

    # 识别解析树中的表名
    def identifyTables(self, tokenList):
        if tokenList.ptype == SUBQUERY:
            self.tableStack.append(False)
        for i in range(len(tokenList.tokens)):
            prevtok = tokenList.tokens[i - 1]
            tok = tokenList.tokens[i]
            if (str(tok) == "." and tok.ttype == sqlparse.tokens.Punctuation and prevtok.ttype == COLUMN):
                prevtok.ttype = TABLE
            elif (str(tok) == "from" and tok.ttype == sqlparse.tokens.Keyword):
                self.tableStack[-1] = True
            elif ((str(tok) == "where" or str(tok) == "on" or str(tok) == "group" or str(tok) == "order" or str(tok) == "union") and tok.ttype == sqlparse.tokens.Keyword):
                self.tableStack[-1] = False
            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyTables(tok)
            elif (tok.ttype == COLUMN):
                if self.tableStack[-1]:
                    tok.ttype = TABLE
        if tokenList.ptype == SUBQUERY:
            self.tableStack.pop()

    # 定义字符串表示函数，将token列表转化为字符串
    def __str__(self):
        return ' '.join([str(tok) for tok in self.tokens])

    # 解析SQL语句，返回token列表的字符串表示
    def parseSql(self):
        return [str(tok) for tok in self.tokens]

# 缩略词处理函数
def revert_abbrev(line):
    # 定义一些常见的缩略词和它们的完整形式的正则表达式
    pat_is = re.compile("(it|he|she|that|this|there|here)(\"s)", re.I)  # 's
    pat_s1 = re.compile("(?<=[a-zA-Z])\"s")  # 's
    pat_s2 = re.compile("(?<=s)\"s?")  # s
    pat_not = re.compile("(?<=[a-zA-Z])n\"t")  # not
    pat_would = re.compile("(?<=[a-zA-Z])\"d")  # would
    pat_will = re.compile("(?<=[a-zA-Z])\"ll")  # will
    pat_am = re.compile("(?<=[I|i])\"m")  # am
    pat_are = re.compile("(?<=[a-zA-Z])\"re")  # are
    pat_ve = re.compile("(?<=[a-zA-Z])\"ve")  # have

    # 使用定义的正则表达式将输入的字符串中的缩略词替换为完整形式
    line = pat_is.sub(r"\1 is", line)
    line = pat_s1.sub("", line)
    line = pat_s2.sub("", line)
    line = pat_not.sub(" not", line)
    line = pat_would.sub(" would", line)
    line = pat_will.sub(" will", line)
    line = pat_am.sub(" am", line)
    line = pat_are.sub(" are", line)
    line = pat_ve.sub(" have", line)

    return line  # 返回处理后的字符串

# 获取词性的函数
def get_wordpos(tag):
    # 如果词性标签以'J'开头，返回形容词
    if tag.startswith('J'):
        return wordnet.ADJ
    # 如果词性标签以'V'开头，返回动词
    elif tag.startswith('V'):
        return wordnet.VERB
    # 如果词性标签以'N'开头，返回名词
    elif tag.startswith('N'):
        return wordnet.NOUN
    # 如果词性标签以'R'开头，返回副词
    elif tag.startswith('R'):
        return wordnet.ADV
    # 如果词性标签不符合以上任何条件，返回None
    else:
        return None

#---------------------子函数1：句子的去冗--------------------

def process_nl_line(line):
    # 句子预处理
    line = revert_abbrev(line)  # 还原缩略词
    line = re.sub('\t+', '\t', line)  # 替换多个制表符为一个
    line = re.sub('\n+', '\n', line)  # 替换多个换行符为一个
    line = line.replace('\n', ' ')  # 将换行符替换为空格
    line = line.replace('\t', ' ')  # 将制表符替换为空格
    line = re.sub(' +', ' ', line)  # 替换多个空格为一个
    line = line.strip()  # 去除行首尾的空格
    line = inflection.underscore(line)  # 将骆驼命名转为下划线命名

    # 去除括号里内容
    space = re.compile(r"\([^\(|^\)]+\)")
    line = re.sub(space, '', line)
    # 去除末尾.和空格
    line = line.strip()
    return line  # 返回处理后的字符串

# 子函数1：句子的分词
def process_sent_word(line):
    # 找单词
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    # 替换小数
    decimal = re.compile(r"\d+(\.\d+)+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换字符串
    string = re.compile(r'\"[^\"]+\"')
    line = re.sub(string, 'TAGSTR', line)
    # 替换十六进制
    decimal = re.compile(r"0[xX][A-Fa-f0-9]+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换数字 56
    number = re.compile(r"\s?\d+\s?")
    line = re.sub(number, ' TAGINT ', line)
    # 替换字符 6c60b8e1
    other = re.compile(r"(?<![A-Z|a-z|_|])\d+[A-Za-z]+")  # 后缀匹配
    line = re.sub(other, 'TAGOER', line)
    cut_words= line.split(' ')
    # 全部小写化
    cut_words = [x.lower() for x in cut_words]
    # 词性标注
    word_tags = pos_tag(cut_words)
    tags_dict = dict(word_tags)
    word_list = []
    for word in cut_words:
        word_pos = get_wordpos(tags_dict[word])
        if word_pos in ['a', 'v', 'n', 'r']:
            # 词性还原
            word = wnler.lemmatize(word, pos=word_pos)
        # 词干提取(效果最好）
        word = wordnet.morphy(word) if wordnet.morphy(word) else word
        word_list.append(word)
    return word_list  # 返回分词后的单词列表

# 去除所有的非法字符
def filter_all_invachar(line):
    # 去除非常用符号；防止解析有误
    line = re.sub('[^(0-9|a-z|A-Z|\-|_|\'|\"|\-|\(|\)|\n)]+', ' ', line)
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line  # 返回处理后的字符串

# 去除部分的非法字符
def filter_part_invachar(line):
    # 去除非常用符号；防止解析有误
    line = re.sub('[^(0-9|a-z|A-Z|\-|#|/|_|,|\'|=|>|<|\"|\-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+', ' ', line)
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line  # 返回处理后的字符串

# 主函数：代码的tokens
def sqlang_code_parse(line):
    line = filter_part_invachar(line)  # 过滤部分非法字符
    line = re.sub('\.+', '.', line)  # 替换多个点为一个
    line = re.sub('\t+', '\t', line)  # 替换多个制表符为一个
    line = re.sub('\n+', '\n', line)  # 替换多个换行符为一个
    line = re.sub(' +', ' ', line)  # 替换多个空格为一个

    line = re.sub('>>+', '', line)  # 新增加，去除多个大于号
    line = re.sub(r"\d+(\.\d+)+",'number',line)  # 新增加，替换小数为'number'

    line = line.strip('\n').strip()  # 去除行首尾的换行和空格
    line = re.findall(r"[\w]+|[^\s\w]", line)  # 找出所有单词和非单词非空白字符
    line = ' '.join(line)  # 将找出的字符用空格连接

    try:
        query = SqlangParser(line, regex=True)  # 创建SQL解析器
        typedCode = query.parseSql()  # 解析SQL语句
        typedCode = typedCode[:-1]  # 去除最后一个字符
        # 骆驼命名转下划线
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')

        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]  # 去除每个token的首尾空格并替换多个空格为一个
        # 全部小写化
        token_list = [x.lower()  for x in cut_tokens]
        # 列表里包含 '' 和' '
        token_list = [x.strip() for x in token_list if x.strip() != '']
        # 返回列表
        return token_list
    # 存在为空的情况，词向量要进行判断
    except:
        return '-1000'

# 主函数：句子的tokens
def sqlang_query_parse(line):
    line = filter_all_invachar(line)  # 过滤所有非法字符
    line = process_nl_line(line)  # 处理句子，去冗
    word_list = process_sent_word(line)  # 句子分词
    # 分完词后,再去掉 括号
    for i in range(0, len(word_list)):
        if re.findall('[\(\)]', word_list[i]):  # 如果单词中包含括号
            word_list[i] = ''  # 将单词设为空
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空
    return word_list  # 返回单词列表

def sqlang_context_parse(line):
    line = filter_part_invachar(line)  # 过滤部分非法字符
    line = process_nl_line(line)  # 处理句子，去冗
    word_list = process_sent_word(line)  # 句子分词
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空
    return word_list  # 返回单词列表


