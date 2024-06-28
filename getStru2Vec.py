# 导入必要的库
import pickle  # 用于序列化和反序列化Python对象
import multiprocessing  # 用于并行处理
from python_structured import *  # 导入自定义的Python解析模块
from sqlang_structured import *  # 导入自定义的SQL解析模块


# 对Python语料中的查询文本进行解析和分词处理
def parse_python_query(data_list):
    """
    对Python语料中的查询文本进行解析和分词处理
    :param data_list: 包含查询文本的列表
    :return: 解析和分词后的查询文本列表
    """
    return [python_query_parse(line) for line in data_list]  # 使用python_query_parse函数解析每一行查询文本


# 对Python语料中的代码文本进行解析和分词处理
def parse_python_code(data_list):
    """
    对Python语料中的代码文本进行解析和分词处理
    :param data_list: 包含代码文本的列表
    :return: 解析和分词后的代码文本列表
    """
    return [python_code_parse(line) for line in data_list]  # 使用python_code_parse函数解析每一行代码文本


# 对Python语料中的前后文文本进行解析和分词处理
def parse_python_context(data_list):
    """
    对Python语料中的前后文文本进行解析和分词处理
    :param data_list: 包含前后文文本的列表
    :return: 解析和分词后的前后文文本列表
    """
    result = []  # 存储解析结果的列表
    for line in data_list:  # 遍历每一行前后文文本
        if line == '-10000':  # 特殊情况处理
            result.append(['-10000'])  # 直接添加特殊标记
        else:
            result.append(python_context_parse(line))  # 使用python_context_parse函数解析前后文文本
    return result  # 返回解析结果


# 对SQL语料中的查询文本进行解析和分词处理
def parse_sql_query(data_list):
    """
    对SQL语料中的查询文本进行解析和分词处理
    :param data_list: 包含查询文本的列表
    :return: 解析和分词后的查询文本列表
    """
    return [sqlang_query_parse(line) for line in data_list]  # 使用sqlang_query_parse函数解析每一行查询文本


# 对SQL语料中的代码文本进行解析和分词处理
def parse_sql_code(data_list):
    """
    对SQL语料中的代码文本进行解析和分词处理
    :param data_list: 包含代码文本的列表
    :return: 解析和分词后的代码文本列表
    """
    return [sqlang_code_parse(line) for line in data_list]  # 使用sqlang_code_parse函数解析每一行代码文本


# 对SQL语料中的上下文文本进行解析和分词处理
def parse_sql_context(data_list):
    """
    对SQL语料中的上下文文本进行解析和分词处理
    :param data_list: 包含上下文文本的列表
    :return: 解析和分词后的上下文文本列表
    """
    result = []  # 存储解析结果的列表
    for line in data_list:  # 遍历每一行上下文文本
        if line == '-10000':  # 特殊情况处理
            result.append(['-10000'])  # 直接添加特殊标记
        else:
            result.append(sqlang_context_parse(line))  # 使用sqlang_context_parse函数解析上下文文本
    return result  # 返回解析结果


# 针对语料，调用上述解析函数进行分词处理，并返回分词结果
def parse_data(data_list, split_num, context_func, query_func, code_func):
    """
    调用解析函数对语料进行分词处理
    :param data_list: 包含语料的列表
    :param split_num: 分割数量
    :param context_func: 解析上下文的函数
    :param query_func: 解析查询的函数
    :param code_func: 解析代码的函数
    :return: 上下文数据、查询数据和代码数据
    """
    pool = multiprocessing.Pool()  # 创建进程池
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]  # 将数据分割成小块

    # 处理上下文数据
    results = pool.map(context_func, split_list)  # 并行调用context_func处理上下文数据
    context_data = [item for sublist in results for item in sublist]  # 展平结果列表
    print(f'上下文条数：{len(context_data)}')  # 打印上下文数据条数

    # 处理查询数据
    results = pool.map(query_func, split_list)  # 并行调用query_func处理查询数据
    query_data = [item for sublist in results for item in sublist]  # 展平结果列表
    print(f'查询条数：{len(query_data)}')  # 打印查询数据条数

    # 处理代码数据
    results = pool.map(code_func, split_list)  # 并行调用code_func处理代码数据
    code_data = [item for sublist in results for item in sublist]  # 展平结果列表
    print(f'代码条数：{len(code_data)}')  # 打印代码数据条数

    pool.close()  # 关闭进程池
    pool.join()  # 等待所有进程完成

    return context_data, query_data, code_data  # 返回上下文数据、查询数据和代码数据


def main(language_type, split_num, source_path, save_path, context_func, query_func, code_func):
    """
    主函数，读取语料文件并调用解析函数进行处理
    :param language_type: 语言类型
    :param split_num: 分割数量
    :param source_path: 源文件路径
    :param save_path: 保存文件路径
    :param context_func: 解析上下文的函数
    :param query_func: 解析查询的函数
    :param code_func: 解析代码的函数
    """
    with open(source_path, 'rb') as file:  # 以二进制读模式打开源文件
        corpus_list = pickle.load(file)  # 读取并反序列化语料列表

    context_data, query_data, code_data = parse_data(corpus_list, split_num, context_func, query_func, code_func)  # 调用parse_data函数解析语料
    query_ids = [item[0] for item in corpus_list]  # 提取语料中的QID

    total_data = [[query_ids[i], context_data[i], code_data[i], query_data[i]] for i in range(len(query_ids))]  # 组合所有数据

    with open(save_path, 'wb') as file:  # 以二进制写模式打开保存文件
        pickle.dump(total_data, file)  # 序列化并保存数据


if __name__ == '__main__':
    # 定义文件路径和保存路径
    staqc_python_path = './ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'

    # 调用main函数处理Python和SQL语料
    main('python', 100, staqc_python_path, staqc_python_save, parse_python_context, parse_python_query, parse_python_code)
    main('sql', 100, staqc_sql_path, staqc_sql_save, parse_sql_context, parse_sql_query, parse_sql_code)

    # 定义大规模语料文件路径和保存路径
    large_python_path = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlabled.pkl'

    large_sql_path = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlabled.pkl'

    # 调用main函数处理大规模Python和SQL语料
    main('python', 100, large_python_path, large_python_save, parse_python_context, parse_python_query, parse_python_code)
    main('sql', 100, large_sql_path, large_sql_save, parse_sql_context, parse_sql_query, parse_sql_code)