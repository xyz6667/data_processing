# 导入必要的库
import pickle  # 用于序列化和反序列化Python对象
from collections import Counter  # 用于计数
import ast  # 用于将字符串解析为Python表达式

def load_pickle(filename):
    """
    从文件中加载Pickle数据
    :param filename: 文件名
    :return: 反序列化后的数据
    """
    with open(filename, 'rb') as f:  # 以二进制读模式打开文件
        data = pickle.load(f, encoding='iso-8859-1')  # 反序列化数据
    return data  # 返回数据

def split_data(total_data, qids):
    """
    根据QID将数据分为单个和多个
    :param total_data: 总数据列表
    :param qids: QID列表
    :return: 单个数据列表和多个数据列表
    """
    result = Counter(qids)  # 统计每个QID出现的次数
    total_data_single = []  # 存储单个数据的列表
    total_data_multiple = []  # 存储多个数据的列表
    for data in total_data:  # 遍历总数据
        if result[data[0][0]] == 1:  # 如果QID只出现一次
            total_data_single.append(data)  # 添加到单个数据列表
        else:
            total_data_multiple.append(data)  # 否则添加到多个数据列表
    return total_data_single, total_data_multiple  # 返回单个数据和多个数据

def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    """
    处理STAQC数据，将其分为单个和多个并保存
    :param filepath: 输入文件路径
    :param save_single_path: 单个数据保存路径
    :param save_multiple_path: 多个数据保存路径
    """
    with open(filepath, 'r') as f:  # 以读模式打开文件
        try:
            total_data = ast.literal_eval(f.read())  # 解析文件内容为Python表达式
        except (SyntaxError, ValueError) as e:  # 捕获解析错误
            print(f"Error parsing file {filepath}: {e}")  # 打印错误信息
            return

    qids = [data[0][0] for data in total_data]  # 提取QID列表
    total_data_single, total_data_multiple = split_data(total_data, qids)  # 分割数据

    with open(save_single_path, "w") as f:  # 以写模式打开单个数据保存文件
        f.write(str(total_data_single))  # 保存单个数据
    with open(save_multiple_path, "w") as f:  # 以写模式打开多个数据保存文件
        f.write(str(total_data_multiple))  # 保存多个数据

def data_large_processing(filepath, save_single_path, save_multiple_path):
    """
    处理大规模数据，将其分为单个和多个并保存
    :param filepath: 输入文件路径
    :param save_single_path: 单个数据保存路径
    :param save_multiple_path: 多个数据保存路径
    """
    total_data = load_pickle(filepath)  # 加载Pickle数据
    qids = [data[0][0] for data in total_data]  # 提取QID列表
    total_data_single, total_data_multiple = split_data(total_data, qids)  # 分割数据

    with open(save_single_path, 'wb') as f:  # 以二进制写模式打开单个数据保存文件
        pickle.dump(total_data_single, f)  # 保存单个数据
    with open(save_multiple_path, 'wb') as f:  # 以二进制写模式打开多个数据保存文件
        pickle.dump(total_data_multiple, f)  # 保存多个数据

def single_unlabeled_to_labeled(input_path, output_path):
    """
    将单个未标记数据转换为标记数据并保存
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    """
    total_data = load_pickle(input_path)  # 加载Pickle数据
    labels = [[data[0], 1] for data in total_data]  # 为每个数据添加标签1
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))  # 按QID和标签排序
    with open(output_path, "w") as f:  # 以写模式打开输出文件
        f.write(str(total_data_sort))  # 保存标记数据

if __name__ == "__main__":
    # 定义STAQC数据文件路径和保存路径
    staqc_python_path = './ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = './ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = './ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)  # 处理Python STAQC数据

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = './ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = './ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)  # 处理SQL STAQC数据

    # 定义大规模数据文件路径和保存路径
    large_python_path = './ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = './ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)  # 处理大规模Python数据

    large_sql_path = './ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = './ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)  # 处理大规模SQL数据

    # 定义单个数据标记文件保存路径
    large_sql_single_label_save = './ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = './ulabel_data/large_corpus/single/python_large_single_label.txt'
    single_unlabeled_to_labeled(large_sql_single_save, large_sql_single_label_save)  # 将SQL单个未标记数据转换为标记数据
    single_unlabeled_to_labeled(large_python_single_save, large_python_single_label_save)  # 将Python单个未标记数据转换为标记数据