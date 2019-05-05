# coding=utf-8
import jieba
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
import sys
from gensim.models import word2vec
import time

def read_rumor(filename):
    """
    :param filename:
    :return: file中的谣言微博的文本内容
    """
    ret = []
    with open(filename, 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            line = line.strip()
            if not len(line):
                continue
            load_dict = json.loads(line)
            ret.append(load_dict['reportedWeibo']['weiboContent'])
        return ret


def read_truth(filename):
    """
    :param filename:
    :return: file中的truth微博的文本内容
    """
    ret = []
    with open(filename, 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            line = line.strip()
            if not len(line):
                continue
            # print(line)
            load_dict = json.loads(line)
            # print(load_dict)
            # print(load_dict['content'])
            ret.append(load_dict['content'])
    return ret


def word_seg(str):
    """
    对str进行jieba中文分词
    :param str:
    :return: 分词后的结果，list形式，去除了空词
    """
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


def word_vectorizer_process(data_set):
    """
    训练词向量模型
    :param data_set:
    :return: 生成词向量模型文件
    """
    vector_size = 32   # 词向量大小
    window_size = 5    # Maximum distance between the current and predicted word within a sentence.
    min_count = 1      # Ignores all words with total frequency lower than this.
    negative_size = 5  # 负采样？
    train_epoch = 30   # 迭代次数
    worker_count = 30  # Use these many worker threads to train the model

    # 制作分完词后的文件，\t隔开每个词
    with open('news_seg_word_data', 'w', encoding='utf-8') as f:
        for temp in data_set:
            for words in temp:
                f.write(words+'\t')

    sentences = word2vec.Text8Corpus(u'news_seg_word_data')
    # print(data_set)
    model = word2vec.Word2Vec(sentences, size=vector_size, window=window_size, min_count=min_count,
                              workers=worker_count, negative=negative_size, iter=train_epoch)
    # print(model.wv[u'座谈会']) 可以通过此方法简单获得一个词的词向量
    model.save(r'news_dim32_ep30.model')


if __name__ == '__main__':
    """
    训练词向量
    1，注意参数在word_vectorizer_process函数中有6个参数可能需要调整
    2，注意文件名,使用不同dataset时，需要更改filename的名字
    3，dataset文件格式一定是每行一个json对象，必要的属性不缺少
    4，文本内容不区分谣言和非谣言，直接作为一个整体的语料训练
    """
    rumor_filename = "data_set/moderate_rumor.json"
    truth_filename = "data_set/moderate_truth.json"

    # 获取微博中文本内容
    content_rumor = read_rumor(rumor_filename)
    content_truth = read_truth(truth_filename)

    content_rumor_small = read_rumor("data_set/small_rumor.json")
    content_truth_small = read_truth("data_set/small_truth.json")

    content_rumor = content_rumor + content_rumor_small
    content_truth = content_truth + content_truth_small

    print("rumor微博个数：%d " % len(content_rumor))
    print("truth微博个数：%d " % len(content_truth))

    # 对文本内容分词，不区分谣言和非谣言，直接作为一个整体的语料训练
    data_set = []
    for str in content_rumor:
        # print(word_seg(str))
        data_set.append(word_seg(str))

    for str in content_truth:
        data_set.append(word_seg(str))
    print("分词完毕。")
    # 训练词向量
    print("开始训练词向量...")
    time_start = time.time()
    word_vectorizer_process(data_set)
    time_end = time.time()
    print("完成词向量训练。用时：%f s" % (time_end - time_start))
