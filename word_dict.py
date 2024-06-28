import pickle

# 获取词汇表
def get_vocab(corpus1, corpus2):
    word_vocab = set()  # 创建一个空集合，用于存储词汇
    for corpus in [corpus1, corpus2]:  # 遍历两个语料库
        for i in range(len(corpus)):  # 遍历语料库中的每个元素
            word_vocab.update(corpus[i][1][0])  # 将元素的第二个子元素的第一个子元素添加到词汇集合中
            word_vocab.update(corpus[i][1][1])  # 将元素的第二个子元素的第二个子元素添加到词汇集合中
            word_vocab.update(corpus[i][2][0])  # 将元素的第三个子元素的第一个子元素添加到词汇集合中
            word_vocab.update(corpus[i][3])  # 将元素的第四个子元素添加到词汇集合中
    print(len(word_vocab))  # 打印词汇集合的大小
    return word_vocab  # 返回词汇集合

# 加载pickle文件
def load_pickle(filename):
    with open(filename, 'rb') as f:  # 打开pickle文件
        data = pickle.load(f)  # 加载pickle文件
    return data  # 返回加载的数据

# 处理词汇
def vocab_processing(filepath1, filepath2, save_path):
    with open(filepath1, 'r') as f:  # 打开第一个文件
        total_data1 = set(eval(f.read()))  # 读取文件内容，并将其转化为集合
    with open(filepath2, 'r') as f:  # 打开第二个文件
        total_data2 = eval(f.read())  # 读取文件内容

    word_set = get_vocab(total_data2, total_data2)  # 获取词汇集合

    excluded_words = total_data1.intersection(word_set)  # 获取第一个集合和词汇集合的交集
    word_set = word_set - excluded_words  # 从词汇集合中移除交集中的元素

    print(len(total_data1))  # 打印第一个集合的大小
    print(len(word_set))  # 打印词汇集合的大小

    with open(save_path, 'w') as f:  # 打开保存路径的文件
        f.write(str(word_set))  # 将词汇集合写入文件

if __name__ == "__main__":
    python_hnn = './data/python_hnn_data_teacher.txt'  # Python HNN数据文件路径
    python_staqc = './data/staqc/python_staqc_data.txt'  # Python STAQC数据文件路径
    python_word_dict = './data/word_dict/python_word_vocab_dict.txt'  # Python词汇字典文件路径

    sql_hnn = './data/sql_hnn_data_teacher.txt'  # SQL HNN数据文件路径
    sql_staqc = './data/staqc/sql_staqc_data.txt'  # SQL STAQC数据文件路径
    sql_word_dict = './data/word_dict/sql_word_vocab_dict.txt'  # SQL词汇字典文件路径

    new_sql_staqc = './ulabel_data/staqc/sql_staqc_unlabled_data.txt'  # 未标注的SQL STAQC数据文件路径
    new_sql_large = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'  # 未标注的大规模SQL数据文件路径
    large_word_dict_sql = './ulabel_data/sql_word_dict.txt'  # 大规模SQL词汇字典文件路径

    final_vocab_processing(sql_word_dict, new_sql_large, large_word_dict_sql)  # 处理词汇