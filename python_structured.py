# -*- coding: utf-8 -*-
import re  # 导入正则表达式模块
import ast  # 导入抽象语法树模块
import sys  # 导入系统模块
import token  # 导入token模块
import tokenize  # 导入tokenize模块

from nltk import wordpunct_tokenize  # 导入NLTK中的单词标点分词器
from io import StringIO  # 导入StringIO模块，用于字符串IO操作
# 骆驼命名法
import inflection  # 导入inflection模块，用于字符串转换

# 词性还原
from nltk import pos_tag  # 导入NLTK中的词性标注函数
from nltk.stem import WordNetLemmatizer  # 导入WordNet词形还原器

wnler = WordNetLemmatizer()  # 创建WordNet词形还原器实例

# 词干提取
from nltk.corpus import wordnet  # 导入WordNet语料库

#############################################################################

# 定义匹配变量赋值模式的正则表达式
PATTERN_VAR_EQUAL = re.compile("(\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)(,\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)*=")
# 定义匹配for循环变量模式的正则表达式
PATTERN_VAR_FOR = re.compile("for\s+[_a-zA-Z][_a-zA-Z0-9]*\s*(,\s*[_a-zA-Z][_a-zA-Z0-9]*)*\s+in")


def repair_program_io(code):
    """
    修复代码中的IO问题
    :param code: 输入代码字符串
    :return: 修复后的代码和代码列表
    """
    # 定义各种正则表达式模式
    pattern_case1_in = re.compile("In ?\[\d+]: ?")  # 匹配输入模式1
    pattern_case1_out = re.compile("Out ?\[\d+]: ?")  # 匹配输出模式1
    pattern_case1_cont = re.compile("( )+\.+: ?")  # 匹配继续模式1

    # 定义各种正则表达式模式
    pattern_case2_in = re.compile(">>> ?")  # 匹配输入模式2
    pattern_case2_cont = re.compile("\.\.\. ?")  # 匹配继续模式2

    # 将所有模式放入列表
    patterns = [pattern_case1_in, pattern_case1_out, pattern_case1_cont,
                pattern_case2_in, pattern_case2_cont]

    # 将代码按行分割
    lines = code.split("\n")
    # 初始化行标记列表
    lines_flags = [0 for _ in range(len(lines))]

    # 初始化代码列表
    code_list = []  # a list of strings

    # 匹配模式
    for line_idx in range(len(lines)):
        line = lines[line_idx]
        for pattern_idx in range(len(patterns)):
            if re.match(patterns[pattern_idx], line):
                lines_flags[line_idx] = pattern_idx + 1
                break
    lines_flags_string = "".join(map(str, lines_flags))

    bool_repaired = False

    # 修复代码
    if lines_flags.count(0) == len(lines_flags):  # 无需修复
        repaired_code = code
        code_list = [code]
        bool_repaired = True

    elif re.match(re.compile("(0*1+3*2*0*)+"), lines_flags_string) or \
            re.match(re.compile("(0*4+5*0*)+"), lines_flags_string):
        repaired_code = ""
        pre_idx = 0
        sub_block = ""
        if lines_flags[0] == 0:
            flag = 0
            while (flag == 0):
                repaired_code += lines[pre_idx] + "\n"
                pre_idx += 1
                flag = lines_flags[pre_idx]
            sub_block = repaired_code
            code_list.append(sub_block.strip())
            sub_block = ""  # 清空

        for idx in range(pre_idx, len(lines_flags)):
            if lines_flags[idx] != 0:
                repaired_code += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

                # 清除子块记录
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

            else:
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

        # 避免遗漏最后一个单元
        if len(sub_block.strip()):
            code_list.append(sub_block.strip())

        if len(repaired_code.strip()) != 0:
            bool_repaired = True

    if not bool_repaired:  # 如果不典型，则仅在每个Out之后移除0标记行。
        repaired_code = ""
        sub_block = ""
        bool_after_Out = False
        for idx in range(len(lines_flags)):
            if lines_flags[idx] != 0:
                if lines_flags[idx] == 2:
                    bool_after_Out = True
                else:
                    bool_after_Out = False
                repaired_code += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

            else:
                if not bool_after_Out:
                    repaired_code += lines[idx] + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

    return repaired_code, code_list


def get_vars(ast_root):
    """
    获取AST树中的变量
    :param ast_root: AST树的根节点
    :return: 变量集合
    """
    return sorted(
        {node.id for node in ast.walk(ast_root) if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)})


def get_vars_heuristics(code):
    """
    使用启发式方法获取代码中的变量
    :param code: 输入代码字符串
    :return: 变量集合
    """
    varnames = set()
    code_lines = [_ for _ in code.split("\n") if len(_.strip())]

    # 最佳努力解析
    start = 0
    end = len(code_lines) - 1
    bool_success = False
    while not bool_success:
        try:
            root = ast.parse("\n".join(code_lines[start:end]))
        except:
            end -= 1
        else:
            bool_success = True
    # print("Best effort parse at: start = %d and end = %d." % (start, end))
    varnames = varnames.union(set(get_vars(root)))
    # print("Var names from base effort parsing: %s." % str(varnames))

    # 处理剩余部分...
    for line in code_lines[end:]:
        line = line.strip()
        try:
            root = ast.parse(line)
        except:
            # 匹配PATTERN_VAR_EQUAL
            pattern_var_equal_matched = re.match(PATTERN_VAR_EQUAL, line)
            if pattern_var_equal_matched:
                match = pattern_var_equal_matched.group()[:-1]  # 移除"="
                varnames = varnames.union(set([_.strip() for _ in match.split(",")]))

            # 匹配PATTERN_VAR_FOR
            pattern_var_for_matched = re.search(PATTERN_VAR_FOR, line)
            if pattern_var_for_matched:
                match = pattern_var_for_matched.group()[3:-2]  # 移除"for"和"in"
                varnames = varnames.union(set([_.strip() for _ in match.split(",")]))

        else:
            varnames = varnames.union(get_vars(root))

    return varnames


def PythonParser(code):
    """
    解析Python代码并提取token
    :param code: 输入代码字符串
    :return: token列表, 变量解析失败标志, token解析失败标志
    """
    bool_failed_var = False
    bool_failed_token = False

    try:
        root = ast.parse(code)
        varnames = set(get_vars(root))
    except:
        repaired_code, _ = repair_program_io(code)
        try:
            root = ast.parse(repaired_code)
            varnames = set(get_vars(root))
        except:
            # failed_var_qids.add(qid)
            bool_failed_var = True
            varnames = get_vars_heuristics(code)

    tokenized_code = []

    def first_trial(_code):
        """
        尝试第一次解析代码
        :param _code: 输入代码字符串
        :return:
        :return: 是否成功解析的布尔值
        """
        if len(_code) == 0:
            return True
        try:
            g = tokenize.generate_tokens(StringIO(_code).readline)
            term = next(g)
        except:
            return False
        else:
            return True

    # 尝试第一次解析代码
    bool_first_success = first_trial(code)
    while not bool_first_success:
        code = code[1:]
        bool_first_success = first_trial(code)
    g = tokenize.generate_tokens(StringIO(code).readline)
    term = next(g)

    bool_finished = False
    while not bool_finished:
        term_type = term[0]
        lineno = term[2][0] - 1
        posno = term[3][1] - 1
        if token.tok_name[term_type] in {"NUMBER", "STRING", "NEWLINE"}:
            tokenized_code.append(token.tok_name[term_type])
        elif not token.tok_name[term_type] in {"COMMENT", "ENDMARKER"} and len(term[1].strip()):
            candidate = term[1].strip()
            if candidate not in varnames:
                tokenized_code.append(candidate)
            else:
                tokenized_code.append("VAR")

        # 获取下一个term
        bool_success_next = False
        while not bool_success_next:
            try:
                term = next(g)
            except StopIteration:
                bool_finished = True
                break
            except:
                bool_failed_token = True
                # print("Failed line: ")
                # print sys.exc_info()
                # 使用wordpunct_tokenizer对错误行进行分词
                code_lines = code.split("\n")
                if lineno > len(code_lines) - 1:
                    print(sys.exc_info())
                else:
                    failed_code_line = code_lines[lineno]  # 错误行
                    # print("Failed code line: %s" % failed_code_line)
                    if posno < len(failed_code_line) - 1:
                        # print("Failed position: %d" % posno)
                        failed_code_line = failed_code_line[posno:]
                        tokenized_failed_code_line = wordpunct_tokenize(failed_code_line)  # 对错误行段进行分词
                        # print("wordpunct_tokenizer tokenization: ")
                        # print(tokenized_failed_code_line)
                        # 追加到之前的分词输出
                        tokenized_code += tokenized_failed_code_line
                    if lineno < len(code_lines) - 1:
                        code = "\n".join(code_lines[lineno + 1:])
                        g = tokenize.generate_tokens(StringIO(code).readline)
                    else:
                        bool_finished = True
                        break
            else:
                bool_success_next = True

    return tokenized_code, bool_failed_var, bool_failed_token

#############################################################################

#############################################################################
# 缩略词处理函数
def revert_abbrev(line):
    """
    还原缩略词
    :param line: 输入文本行
    :return: 还原后的文本行
    """
    pat_is = re.compile("(it|he|she|that|this|there|here)(\"s)", re.I)
    # 's
    pat_s1 = re.compile("(?<=[a-zA-Z])\"s")
    # s
    pat_s2 = re.compile("(?<=s)\"s?")
    # not
    pat_not = re.compile("(?<=[a-zA-Z])n\"t")
    # would
    pat_would = re.compile("(?<=[a-zA-Z])\"d")
    # will
    pat_will = re.compile("(?<=[a-zA-Z])\"ll")
    # am
    pat_am = re.compile("(?<=[I|i])\"m")
    # are
    pat_are = re.compile("(?<=[a-zA-Z])\"re")
    # have
    pat_ve = re.compile("(?<=[a-zA-Z])\"ve")

    line = pat_is.sub(r"\1 is", line)
    line = pat_s1.sub("", line)
    line = pat_s2.sub("", line)
    line = pat_not.sub(" not", line)
    line = pat_would.sub(" would", line)
    line = pat_will.sub(" will", line)
    line = pat_am.sub(" am", line)
    line = pat_are.sub(" are", line)
    line = pat_ve.sub(" have", line)

    return line


# 获取词性
def get_wordpos(tag):
    """
    获取词性
    :param tag: 词性标签
    :return: 对应的WordNet词性
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# ---------------------子函数1：句子的去冗--------------------
def process_nl_line(line):
    """
    预处理自然语言句子，去除冗余
    :param line: 输入句子
    :return: 处理后的句子
    """
    # 句子预处理
    line = revert_abbrev(line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = line.replace('\n', ' ')
    line = re.sub(' +', ' ', line)
    line = line.strip()
    # 骆驼命名转下划线
    line = inflection.underscore(line)

    # 去除括号里内容
    space = re.compile(r"\([^(|^)]+\)")  # 后缀匹配
    line = re.sub(space, '', line)
    # 去除开始和末尾空格
    line = line.strip()
    return line


# ---------------------子函数2：句子的分词--------------------
def process_sent_word(line):
    """
    对句子进行分词处理
    :param line: 输入句子
    :return: 分词后的单词列表
    """
    # 找单词
    line = re.findall(r"\w+|[^\s\w]", line)
    line = ' '.join(line)
    # 替换小数
    decimal = re.compile(r"\d+(\.\d+)+")
    line = re.sub(decimal, 'TAGINT')
    # 替换字符串
    string = re.compile(r'\"[^\"]+\"')
    line = re.sub(string, 'TAGSTR')
    # 替换十六进制
    decimal = re.compile(r"0[xX][A-Fa-f0-9]+")
    line = re.sub(decimal, 'TAGINT')
    # 替换数字
    number = re.compile(r"\s?\d+\s?")
    line = re.sub(number, ' TAGINT ')
    # 替换字符
    other = re.compile(r"(?<![A-Z|a-z_])\d+[A-Za-z]+")  # 后缀匹配
    line = re.sub(other, 'TAGOER', line)
    cut_words = line.split(' ')
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
    return word_list

#############################################################################

def filter_all_invachar(line):
    """
    去除所有非常用符号
    :param line: 输入文本行
    :return: 处理后的文本行
    """
    # 去除非常用符号；防止解析有误
    assert isinstance(line, object)
    line = re.sub('[^(0-9|a-zA-Z\-_\'\")\n]+', ' ', line)
    # 包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line


def filter_part_invachar(line):
    """
    去除部分非常用符号
    :param line: 输入文本行
    :return: 处理后的文本行
    """
    # 去除非常用符号；防止解析有误
    line = re.sub('[^(0-9|a-zA-Z\-_\'\")\n]+', ' ', line)
    # 包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line


########################主函数：代码的tokens#################################
def python_code_parse(line):
    """
    解析Python代码并提取tokens
    :param line: 输入代码行
    :return: token列表
    """
    line = filter_part_invachar(line)
    line = re.sub('\.+', '.', line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = re.sub('>>+', '', line)  # 新增加的处理
    line = re.sub(' +', ' ', line)
    line = line.strip('\n').strip()
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    try:
        typedCode, failed_var, failed_token = PythonParser(line)
        # 骆驼命名转下划线
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')

        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
        # 全部小写化
        token_list = [x.lower() for x in cut_tokens]
        # 列表里包含 '' 和' '
        token_list = [x.strip() for x in token_list if x.strip() != '']
        return token_list
        # 存在为空的情况，词向量要进行判断
    except:
        return '-1000'


########################主函数：代码的tokens#################################


#######################主函数：句子的tokens##################################

def python_query_parse(line):
    """
    解析自然语言查询并提取tokens
    :param line: 输入查询句子
    :return: token列表
    """
    line = filter_all_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    # 分完词后,再去掉括号
    for i in range(0, len(word_list)):
        if re.findall('[()]', word_list[i]):
            word_list[i] = ''
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空

    return word_list


def python_context_parse(line):
    """
    解析自然语言上下文并提取tokens
    :param line: 输入上下文句子
    :return: token列表
    """
    line = filter_part_invachar(line)
    # 在这一步的时候驼峰命名被转换成了下划线
    line = process_nl_line(line)
    print(line)
    word_list = process_sent_word(line)
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空
    return word_list


#######################主函数：句子的tokens##################################
if __name__ == '__main__':
    # 测试自然语言查询解析
    print(python_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    # 输出：['change', 'row', 'height', 'and', 'column', 'width', 'in', 'libreoffice', 'calc', 'use', 'python', 'tagint']
    print(python_query_parse('What is the standard way to add N seconds to datetime.time in Python?'))
    # 输出：['what', 'is', 'the', 'standard', 'way', 'to', 'add', 'n', 'second', 'to', 'datetime', 'time', 'in', 'python']
    print(python_query_parse("Convert INT to VARCHAR SQL 11?"))
    # 输出：['convert', 'int', 'to', 'varchar', 'sql', 'tagint']
    print(python_query_parse(
        'python construct a dictionary {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 2], 3: [0, 0, 3], ...,999: [9, 9, 9]}'))
    # 输出：['python', 'construct', 'a', 'dictionary', 'tagint', 'tagint', 'tagint', 'tagint', 'tagint', 'tagint', 'tagint', 'tagint', 'tagint', 'tagint', 'tagint', 'tagint', 'tagint']
    # 测试自然语言上下文解析
    print(python_context_parse(
        'How to calculateAnd the value of the sum of squares defined as \n 1^2 + 2^2 + 3^2 + ... +n2 until a user specified sum has been reached sql()'))
    # 输出：['how', 'to', 'calculateand', 'the', 'value', 'of', 'the', 'sum', 'of', 'square', 'defined', 'as', 'tagint', 'tagint', 'tagint', 'tagint', 'until', 'a', 'user', 'specified', 'sum', 'has', 'been', 'reached', 'sql']
    print(python_context_parse('how do i display records (containing specific) information in sql() 11?'))
    # 输出：['how', 'do', 'i', 'display', 'record', 'containing', 'specific', 'information', 'in', 'sql', 'tagint']
    print(python_context_parse('Convert INT to VARCHAR SQL 11?'))
    # 输出：['convert', 'int', 'to', 'varchar', 'sql', 'tagint']
    # 测试代码解析
    print(python_code_parse(
        'if(dr.HasRows)\n{\n // ....\n}\nelse\n{\n MessageBox.Show("ReservationAnd Number Does Not Exist","Error", MessageBoxButtons.OK, MessageBoxIcon.Asterisk);\n}'))
    # 输出：['if', '(', 'var', ')', '{', '}', 'else', '{', 'var', '(', 'string', ',', 'string', ',', 'var', '.', 'var', ',', 'var', '.', 'var', ')', ';', '}']
    print(python_code_parse('root -> 0.0 \n while root_ * root < n: \n root = root + 1 \n print(root * root)'))
    # 输出：['root', '->', 'number', 'while', 'var', '*', 'var', '<', 'var', ':', 'var', '=', 'var', '+', 'number', 'print', '(', 'var', '*', 'var', ')']
    print(python_code_parse('root = 0.0 \n while root * root < n: \n print(root * root) \n root = root + 1'))
    # 输出：['var', '=', 'number', 'while', 'var', '*', 'var', '<', 'var', ':', 'print', '(', 'var', '*', 'var', ')', 'var', '=', 'var', '+', 'number']
    print(python_code_parse('n = 1 \n while n <= 100: \n n = n + 1 \n if n > 10: \n  break print(n)'))
    # 输出：['var', '=', 'number', 'while', 'var', '<=', 'number', ':', 'var', '=', 'var', '+', 'number', 'if', 'var', '>', 'number', ':', 'break', 'print', '(', 'var', ')']
    print(python_code_parse(
        "diayong(2) def sina_download(url, output_dir='.', merge=True, info_only=False, **kwargs):\n    if 'news.sina.com.cn/zxt' in url:\n        sina_zxt(url, output_dir=output_dir, merge=merge, info_only=info_only, **kwargs)\n  return\n\n    vid = match1(url, r'vid=(\\d+)')\n    if vid is None:\n        video_page = get_content(url)\n        vid = hd_vid = match1(video_page, r'hd_vid\\s*:\\s*\\'([^\\']+)\\'')\n  if hd_vid == '0':\n            vids = match1(video_page, r'[^\\w]vid\\s*:\\s*\\'([^\\']+)\\'').split('|')\n            vid = vids[-1]\n\n    if vid is None:\n        vid = match1(video_page, r'vid:\"?(\\d+)\"?')\n    if vid:\n   sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n    else:\n        vkey = match1(video_page, r'vkey\\s*:\\s*\"([^\"]+)\"')\n        if vkey is None:\n            vid = match1(url, r'#(\\d+)')\n            sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n            return\n        title = match1(video_page, r'title\\s*:\\s*\"([^\"]+)\"')\n        sina_download_by_vkey(vkey, title=title, output_dir=output_dir, merge=merge, info_only=info_only)"))
    # 输出：['diayong', '(', 'number', ')', 'def', 'var', '(', 'var', ',', 'var', '=', 'string', ',', 'var', '=', 'var', ',', 'var', '=', 'var', ',', 'var', '=', 'var', ',', '**', 'var', ')', ':', 'if', 'string', 'in', 'var', ':', 'var', '(', 'var', ',', 'var', '=', 'var', ',', 'var', '=', 'var', ',', 'var', '=', 'var', ',', '**', 'var', ')', 'return', 'var', '=', 'var', '(', 'var', ',', 'string', ')', 'if', 'var', 'is', 'var', ':', 'var', '=', 'var', '(', 'var', ')', 'var', '=', 'var', '=', 'var', '(', 'var', ',', 'string', ')', 'if', 'var', '==', 'string', ':', 'var', '=', 'var', '(', 'var', ',', 'string', ')', '.', 'split', '(', 'string', ')', 'var', '=', 'var', '[', '-', 'number', ']', 'if', 'var', 'is', 'var', ':', 'var', '=', 'var', '(', 'var', ',', 'string', ')', 'if', 'var', ':', 'var', '(', 'var', ',', 'var', '=', 'var', ',', 'var', '=', 'var', ',', 'var', '=', 'var', ')', 'else', ':', 'var', '=', 'var', '(', 'var', ',', 'string', ')', 'if', 'var', 'is', 'var', ':', 'var', '=', 'var', '(', 'var', ',', 'string', ')', 'var', '(', 'var', ',', 'var', '=', 'var', ',', 'var', '=', 'var', ',', 'var', '=', 'var', ')', 'return', 'title', '=', 'var', '(', 'var', ',', 'string', ')', 'var', '(', 'var', ',', 'var', '=', 'var', ',', 'var', '=', 'var', ',', 'var', '=', 'var', ')']
    print(python_code_parse("d = {'x': 1, 'y': 2, 'z': 3} \n for key in d: \n  print (key, 'corresponds to', d[key])"))
    # 输出：['var', '=', '{', 'string', ':', 'number', ',', 'string', ':', 'number', ',', 'string', ':', 'number', '}', 'for', 'var', 'in', 'var', ':', 'print', '(', 'var', ',', 'string', ',', 'var', '[', 'var', ']', ')']
    print(python_code_parse(
        '  #       page  hour  count\n # 0     3727441     1   2003\n # 1     3727441     2    654\n # 2     3727441     3   5434\n # 3     3727458     1    326\n # 4     3727458     2   2348\n # 5     3727458     3   4040\n # 6   3727458_1     4    374\n # 7   3727458_1     5   2917\n # 8   3727458_1     6   3937\n # 9     3735634     1   1957\n # 10    3735634     2   2398\n # 11    3735634     3   2812\n # 12    3768433     1    499\n # 13    3768433     2   4924\n # 14    3768433     3   5460\n # 15  3768433_1     4   1710\n # 16  3768433_1     5   3877\n # 17  3768433_1     6   1912\n # 18  3768433_2     7   1367\n # 19  3768433_2     8   1626\n # 20  3768433_2     9   4750\n'))

