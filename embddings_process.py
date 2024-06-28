# 导入所需的库
import pickle
import numpy as np
from gensim.models import KeyedVectors

# 将词向量文件保存为二进制文件
def trans_bin(path1, path2):
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)  # 从文本中加载词向量
    wv_from_text.init_sims(replace=True)  # 预计算L2-norms
    wv_from_text.save(path2)  # 保存词向量为二进制文件

# 构建新的词典和词向量矩阵
def get_new_dict(type_vec_path, type_word_path, final_vec_path, final_word_path):
    model = KeyedVectors.load(type_vec_path, mmap='r')  # 加载词向量模型
    with open(type_word_path, 'r') as f:  # 打开词典文件
        total_word = eval(f.read())  # 读取词典

    word_dict = ['PAD', 'SOS', 'EOS', 'UNK']  # 设置特殊词
    fail_word = []  # 用于保存无法找到词向量的词
    rng = np.random.RandomState(None)  # 创建随机数生成器
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()  # 为PAD生成零向量
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()  # 为UNK生成随机向量
    sos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()  # 为SOS生成随机向量
    eos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()  # 为EOS生成随机向量
    word_vectors = [pad_embedding, sos_embedding, eos_embedding, unk_embedding]  # 初始化词向量列表

    for word in total_word:  # 遍历词典中的每个词
        try:
            word_vectors.append(model.wv[word])  # 尝试获取词的向量并添加到列表中
            word_dict.append(word)  # 将词添加到词典中
        except:
            fail_word.append(word)  # 如果无法获取词的向量，则将词添加到fail_word列表中

    word_vectors = np.array(word_vectors)  # 将词向量列表转换为numpy数组
    word_dict = dict(map(reversed, enumerate(word_dict)))  # 创建词典，键为词，值为索引

    with open(final_vec_path, 'wb') as file:  # 打开最终的词向量文件
        pickle.dump(word_vectors, file)  # 将词向量保存为pickle文件

    with open(final_word_path, 'wb') as file:  # 打开最终的词典文件
        pickle.dump(word_dict, file)  # 将词典保存为pickle文件

    print("完成")  # 打印完成信息


# 得到词在词典中的位置
def get_index(type, text, word_dict):
    location = []
    if type == 'code':  # 如果类型为代码
        location.append(1)  # 添加起始标记
        len_c = len(text)  # 获取代码长度
        if len_c + 1 < 350:  # 如果代码长度小于350
            if len_c == 1 and text[0] == '-1000':  # 如果代码只有一个元素且为-1000
                location.append(2)  # 添加结束标记
            else:  # 否则
                for i in range(0, len_c):  # 遍历代码中的每个元素
                    index = word_dict.get(text[i], word_dict['UNK'])  # 获取元素在词典中的索引，如果不存在则返回UNK的索引
                    location.append(index)  # 将索引添加到位置列表中
                location.append(2)  # 添加结束标记
        else:  # 如果代码长度大于等于350
            for i in range(0, 348):  # 遍历前348个元素
                index = word_dict.get(text[i], word_dict['UNK'])  # 获取元素在词典中的索引，如果不存在则返回UNK的索引
                location.append(index)  # 将索引添加到位置列表中
            location.append(2)  # 添加结束标记
    else:  # 如果类型为文本
        if len(text) == 0:  # 如果文本为空
            location.append(0)  # 添加PAD标记
        elif text[0] == '-10000':  # 如果文本的第一个元素为-10000
            location.append(0)  # 添加PAD标记
        else:  # 否则
            for i in range(0, len(text)):  # 遍历文本中的每个元素
                index = word_dict.get(text[i], word_dict['UNK'])  # 获取元素在词典中的索引，如果不存在则返回UNK的索引
                location.append(index)  # 将索引添加到位置列表中

    return location  # 返回位置列表

# 将训练、测试、验证语料序列化
# 查询：25 上下文：100 代码：350
def serialization(word_dict_path, type_path, final_type_path):
    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)  # 从pickle文件中加载词典

    with open(type_path, 'r') as f:
        corpus = eval(f.read())  # 读取语料

    total_data = []  # 初始化数据列表

    for i in range(len(corpus)):  # 遍历语料中的每个元素
        qid = corpus[i][0]  # 获取问题ID

        Si_word_list = get_index('text', corpus[i][1][0], word_dict)  # 获取上下文的索引列表
        Si1_word_list = get_index('text', corpus[i][1][1], word_dict)  # 获取上下文的索引列表
        tokenized_code = get_index('code', corpus[i][2][0], word_dict)  # 获取代码的索引列表
        query_word_list = get_index('text', corpus[i][3], word_dict)  # 获取查询的索引列表
        block_length = 4  # 设置块长度
        label = 0  # 设置标签

        # 如果索引列表的长度大于指定长度，则截取前面的部分，否则在后面补充PAD标记
        Si_word_list = Si_word_list[:100] if len(Si_word_list) > 100 else Si_word_list + [0] * (100 - len(Si_word_list))
        Si1_word_list = Si1_word_list[:100] if len(Si1_word_list) > 100 else Si1_word_list + [0] * (100 - len(Si1_word_list))
        tokenized_code = tokenized_code[:350] + [0] * (350 - len(tokenized_code))
        query_word_list = query_word_list[:25] if len(query_word_list) > 25 else query_word_list + [0] * (25 - len(query_word_list))

        one_data = [qid, [Si_word_list, Si1_word_list], [tokenized_code], query_word_list, block_length, label]  # 创建一个数据元素
        total_data.append(one_data)  # 将数据元素添加到数据列表中

    with open(final_type_path, 'wb') as file:  # 打开最终的语料文件
        pickle.dump(total_data, file)  # 将数据列表保存为pickle文件


if __name__ == '__main__':
    # 词向量文件路径
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin'  # Python词向量文件路径
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'  # SQL词向量文件路径

    # ==========================最初基于Staqc的词典和词向量==========================

    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'  # Python词典文件路径
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'  # Python词向量文件路径
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'  # Python词典文件路径

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'  # SQL词典文件路径
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'  # SQL词向量文件路径
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'  # SQL词典文件路径

    # 这两行代码用于创建新的词典和词向量矩阵，但目前被注释掉了
    # get_new_dict(ps_path_bin, python_word_path, python_word_vec_path, python_word_dict_path)
    # get_new_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

    # =======================================最后打标签的语料========================================

    # sql 待处理语料地址
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'  # 待处理的SQL语料文件路径
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'  # 待处理的大规模SQL语料文件路径
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'  # 大规模SQL词典文件路径

    # sql最后的词典和对应的词向量
    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'  # 最后的SQL词向量文件路径
    sqlfinal_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'  # 最后的SQL词典文件路径

    # 这两行代码用于创建新的词典和词向量矩阵，但目前被注释掉了
    # get_new_dict(sql_path_bin, final_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)
    # get_new_dict_append(sql_path_bin, sql_word_dict_path, sql_word_vec_path, large_word_dict_sql, sql_final_word_vec_path,sql_final_word_dict_path)

    staqc_sql_f = '../hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'  # 未标注的SQL语料文件路径
    large_sql_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'  # 未标注的大规模SQL语料文件路径

    # 这两行代码用于序列化语料，但目前被注释掉了
    # Serialization(sql_final_word_dict_path, new_sql_staqc, staqc_sql_f)
    # Serialization(sql_final_word_dict_path, new_sql_large, large_sql_f)

    # python
    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'  # 待处理的Python语料文件路径
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'  # 待处理的大规模Python语料文件路径
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'  # Python词典文件路径
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'  # 大规模Python词典文件路径

    # python最后的词典和对应的词向量
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'  # 最后的Python词向量文件路径
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'  # 最后的Python词典文件路径

    # 这两行代码用于创建新的词典和词向量矩阵，但目前被注释掉了
    # get_new_dict(ps_path_bin, final_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)
    # get_new_dict_append(ps_path_bin, python_word_dict_path, python_word_vec_path, large_word_dict_python, python_final_word_vec_path,python_final_word_dict_path)

    # 处理成打标签的形式
    staqc_python_f = '../hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'  # 未标注的Python语料文件路径
    large_python_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_python_large_multiple_unlable.pkl'  # 未标注的大规模Python语料文件路径

    # 这行代码用于序列化语料，但目前被注释掉了
    # Serialization(python_final_word_dict_path, new_python_staqc, staqc_python_f)
    serialization(python_final_word_dict_path, new_python_large, large_python_f)  # 序列化大规模Python语料

    print('序列化完毕')  # 打印完成信息
    # test2(test_python1,test_python2,python_final_word_dict_path,python_final_word_vec_path)  # 这行代码用于测试