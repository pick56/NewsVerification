# coding=utf-8
import jieba
import json
from gensim.models import word2vec
from langconv import *
import numpy
from keras.utils import np_utils


word_size = 100           # 每个微博取多少个词
word_dim = 64             # 词向量的维度
social_features_dim = 16  # social feature维度
nb_epoch = 1000           # epoch训练次数
batch_size = 100          # 批大小
input_dim = word_dim


def word_seg(str):
    ret = []
    seg_list = jieba.cut(str, cut_all=False)
    # print(seg_list)
    # ret.append(" ".join(seg_list))
    # ret = list(seg_list)
    for word in seg_list:
        temp = word.strip()
        if len(temp) <= 0:
            continue
        ret.append(temp)
    return ret


def process(str):
    replace_terms = [' ', '...', '展开全文c', '谣言', '辟谣', 'quot', 'nbsp']
    for term in replace_terms:
        str = str.replace(term, ' ')
        content1 = str.strip()
        content1 = Converter('zh-hans').convert(content1)
    return content1


def get_text_features(filename, num, dim):
    # w2v_model = word2vec.Word2Vec.load(r'wiki-corpus/wiki_corpus.model')
    w2v_model = word2vec.Word2Vec.load(r'news_dim64_ep30.model')

    ret = []
    with open(filename, 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            # 循环每一个微博
            line = line.strip()
            if not len(line):
                print(line)
                print("some thing wrong")
                continue
            text = process(line)  # 去除停用词，繁体字转化
            # print(text)
            text_seg = word_seg(text)  # 分词
            # print(text_seg)
            st = []
            it = 1
            for word in text_seg:
                if it > num:
                    break
                it = it + 1
                try:
                    vec = w2v_model.wv[word].tolist()
                except:
                    vec = [0 for x in range(0, dim)]
                st.append(vec)
            for i in range(len(st), num):
                st.append([0 for x in range(0, dim)])
            ret.append(st)
        # print(len(ret))
    return ret


def get_train_data(rumor_filename, truth_filename):
    # 得到数据集每微博的text features
    all_text_features = get_text_features(rumor_filename, word_size, word_dim)
    num1 = len(all_text_features)
    all_text_features = all_text_features + get_text_features(truth_filename, word_size, word_dim)
    num2 = len(all_text_features) - num1

    print("提取了text features，%d 个rumor %d个truth" % (num1, num2))
    print("转换为训练数据")

    data_x = []
    for feature in all_text_features:  # 对每个微博
        temp_x = []
        for temp in feature:  # 100个词
            temp_x.append(numpy.array(temp))
        data_x.append(numpy.array(temp_x))
    dx = numpy.array(data_x)
    X = dx.reshape(num1 + num2, word_size, input_dim)
    # print(X.shape)

    # 准备标签数据
    print("准备标签数据")
    lables = [1 for i in range(0, num1)] + [0 for i in range(0, num2)]
    # print(lables)
    # print(len(lables))

    y = numpy.array(lables)
    # 将数据变成one-hot形式
    y = np_utils.to_categorical(y)
    # print(y)
    print("标签数据准备完毕")
    return X, y


if __name__ == '__main__':
    """
    将原始文本转化成numpy数组
    """

    import os.path

    train_X_dir = "npy/train_text_X.npy"
    train_y_dir = "npy/train_text_y.npy"
    val_X_dir = "npy/val_text_X.npy"
    val_y_dir = "npy/val_text_y.npy"

    # 训练集合
    train_X, train_y = get_train_data("data_set/train_rumor_text.txt", "data_set/train_truth_text.txt")
    # 验证集
    val_X, val_y = get_train_data("data_set/test_rumor_text.txt", "data_set/test_truth_text.txt")
    # numpy.save(train_X_dir, train_X)
    # numpy.save(train_y_dir, train_y)
    numpy.save(val_X_dir, val_X)
    numpy.save(val_y_dir, val_y)
    print("训练npy数据已保存")
