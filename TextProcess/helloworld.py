# coding=utf-8
import jieba
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
import sys
from gensim.models import word2vec

def read_rumor(filename):
    ret = []
    with open(filename, 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            line = line.strip()
            if not len(line):
                continue
            # print(line)
            load_dict = json.loads(line)
            # print(load_dict)
            # print(load_dict['reportedWeibo']['weiboContent'])
            ret.append(load_dict['reportedWeibo']['weiboContent'])
        return ret


def read_truth(filename):
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
    ret = []
    seg_list = jieba.cut(str, cut_all=False)
    # print(seg_list)
    # ret.append(" ".join(seg_list))
    ret = list(seg_list)
    return ret


def count_words(str, words):
    ret = 0
    for word in words:
        ret += str.count(word)
    return ret


def word_vectorizer_process(data_set):
    '''
    word2vec
    '''
    vector_size = 32   # 词向量大小
    window_size = 5    # Maximum distance between the current and predicted word within a sentence.
    min_count = 1      # Ignores all words with total frequency lower than this.
    negative_size = 5  # 负采样？
    train_epoch = 30   # 迭代次数
    worker_count = 30  # Use these many worker threads to train the model

    with open('dataset.txt', 'w', encoding='utf-8') as f:
        for temp in data_set:
            for words in temp:
                f.write(words+'\t')

    sentences = word2vec.Text8Corpus(u'dataset.txt')
    # print(data_set)
    model = word2vec.Word2Vec(sentences, size=vector_size, window=window_size, min_count=min_count,
                              workers=worker_count, negative=negative_size, iter=train_epoch)
    # print(model.wv[u'座谈会'])
    # print(model.wv['#'])
    # model.train([[u"座谈会", u"会议"]], total_examples=1, epochs=1)
    # print(model.wv[u"座谈会"])
    model.save(r'news_dim32_ep30.model')


def get_social_features_rumor(filename):

    """
    给一个微博的json文件，可以获得所有的特征
    统计以下特征16个
    2Number of exclamation/question mark
    2Number of words/characters
    2Number of positive/negative words
    3Number of first/second/third order of pronoun
    3Number of URL/@/#
    3Number of People/Location/Organization
    1Sentiment score

    number_of_exclamation_mark = 0
    number_of_question_mark = 0
    number_of_words = 0
    number_of_characters = 0
    number_of_positive_words = 0
    number_of_negative_words = 0
    number_of_first_pronoun = 0
    number_of_second_pronoun = 0
    number_of_third_pronoun = 0
    number_of_url = 0
    number_of_at = 0
    number_of_num = 0
    number_of_people = 0
    number_of_location = 0
    number_of_organization = 0
    Sentiment_score = 0
    """


    ret = []
    with open(filename, 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            # 循环每一个微博
            all_social_features = []
            line = line.strip()
            if not len(line):
                continue
            # print(line)
            load_dict = json.loads(line)
            # print(load_dict)
            # print(load_dict['reportedWeibo']['weiboContent'])
            # ret.append(load_dict['reportedWeibo']['weiboContent'])
            text = load_dict['reportedWeibo']['weiboContent']
            # print(text)
            all_social_features.append(text.count('!')+text.count('！'))  # number_of_exclamation_mark
            all_social_features.append(text.count('?') + text.count('？'))  # number_of_question_mark
            all_social_features.append(len(word_seg(text)))  # number_of_words
            all_social_features.append(len(text))  # number_of_characters
            all_social_features.append(0)  # number_of_positive_words
            all_social_features.append(0)  # number_of_negative_words
            all_social_features.append(0)  # number_of_first_pronoun
            all_social_features.append(0)  # number_of_second_pronoun
            all_social_features.append(0)  # number_of_third_pronoun
            all_social_features.append(text.count("https://")+text.count("http://"))  # number_of_url
            all_social_features.append(text.count('@'))  # number_of_at
            all_social_features.append(text.count('#'))  # number_of_num
            all_social_features.append(0)  # number_of_people
            all_social_features.append(0)  # number_of_location
            all_social_features.append(0)  # number_of_organization
            # print(all_social_features)
            ret.append(all_social_features)
    return ret


if __name__ == '__main__':

    '''
    content_rumor = read_rumor("smallrumor.json")
    content_truth = read_truth("smalltruth.json")

    print(len(content_rumor))
    print(content_rumor)
    print(len(content_truth))
    print(content_truth)

    data_set = []

    for str in content_rumor:
        # print(word_seg(str))
        data_set.append(word_seg(str))

    for str in content_truth:
        data_set.append(word_seg(str))

    print(len(data_set))
    # print(data_set[0])

    word_vectorizer_process(data_set)

    print("ok")
    '''

    # w2v_model = word2vec.Word2Vec.load(r'news_dim32_ep30.model')
    # print(w2v_model.wv[u'座谈会'])

    # all_social_features = get_social_features_rumor("smallrumor.json")
    # print(all_social_features)
    # get_social_features_truth("smalltruth.json")
    test = "详情：https://weibo.com/5921199319/H0P0iBgh8详情：https://weibo.com/5921199319/H0P0iBgh8"
    lists = ['t', 'p']
    print(count_words(test, lists))




# if __name__ == '__main__' and sys.argv[1]=='train_word2vector':


# 第一步读取json文件，只保留微博内容部分？
'''
问题：
1，truth中21个属性

2，rumor17个属性，10个是微博的，7个是用户的

3，都有的共同属性14-1个，其中1个不确定

4，哪个部分是textual feature
微博内容，截断前70个字符
得到70个32维度的向量

5，social context feature
得到1个16维的向量

6，1个16维的向量全连接变成1个32维的向量
7，一个微博的textual feature和social textual feature就是71个32维的向量
8，然后用71个cell的lstm，跑一个many2many得到71个32维的向量，平均一下就是一个32维的向量
9，然后和图片融合，最基本的融合就是图片的32维和文本的32维拼接在一起，变成64维度的东西
10，然后全连接层，做一个softmax二分类
'''