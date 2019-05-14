# coding=utf-8
import jieba
import pickle
import json
import sys
from gensim.models import word2vec
import time


def word_vectorizer_process():
    """
    训练词向量模型
    :param data_set:
    :return: 生成词向量模型文件
    """
    vector_size = 64   # 词向量大小
    window_size = 5    # Maximum distance between the current and predicted word within a sentence.
    min_count = 1      # Ignores all words with total frequency lower than this.
    negative_size = 5  # 负采样？
    train_epoch = 30   # 迭代次数
    worker_count = 30  # Use these many worker threads to train the model

    sentences = word2vec.Text8Corpus(u'all_data_seg.txt')
    # print(data_set)
    model = word2vec.Word2Vec(sentences, size=vector_size, window=window_size, min_count=min_count,
                              workers=worker_count, negative=negative_size, iter=train_epoch)
    # print(model.wv[u'座谈会']) 可以通过此方法简单获得一个词的词向量
    model.save(r'news_dim64_ep30.model')


if __name__ == '__main__':

    print("开始训练词向量...")
    time_start = time.time()
    word_vectorizer_process()
    time_end = time.time()
    print("完成词向量训练。用时：%f s" % (time_end - time_start))
