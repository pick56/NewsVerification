# coding=utf-8
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import sys
from gensim.models import word2vec

import numpy
import time
from keras.utils import np_utils
import os.path

import senti_score

word_size = 100           # 每个微博取多少个词
word_dim = 64             # 词向量的维度
social_dim = 16+6+3       # 25的social_feature维度


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


def find_in_dic(filename, text):
    """
    微博文本在某类词典中出现个数
    :param filename: 词典文件名
    :param text: 分完词后的微博文本
    :return: 出现个数
    """
    ret = 0
    # 微博中每个词在词典中是否出现
    dic_list = []
    with open(filename, 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            line = line.strip()
            if not len(line):
                continue
            dic_list.append(line)

    # 循环每个微博中的词
    # print(text)
    for word in text:
        word = word.strip()
        # print(word)
        if not len(word):
            continue
        # 循环每个词典中的词
        for line in dic_list:
            line = line.strip()
            if not len(line):
                continue
            # 匹配了
            # print(word+" "+line)
            if word == line:
                ret = ret + 1
                break
    return ret


def get_ner(text):
    """
    输入未分词的文本，桉树徐输出人名、地名、组织名数目
    按照ictcls的定义
    nr人名
    nr1 汉语姓氏
    nr2 汉语名字
    nrj 日语人名
    nrf 音译人名

    ns 地名
    nsf 音译地名

    nt 机构团体名
    :param text:
    :return: 三元组
    """
    people = 0
    local = 0
    org = 0
    words = pseg.cut(text)
    for word, flag in words:
        # print(word, flag)
        if flag[0:2] == "nr":
            people = people + 1
        if flag[0:2] == "ns":
            local = local + 1
        if flag[0:2] == "nt":
            org = org + 1
    return people, local, org


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
    """
    ret = []
    it = 1
    with open(filename, 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            if it % 1000 == 0:
                print(it)
                # break
            it = it + 1
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
            all_social_features.append(find_in_dic("dic/emotion_dic/positive.txt", word_seg(text)))  # number_of_positive_words
            all_social_features.append(find_in_dic("dic/emotion_dic/negative.txt", word_seg(text)))  # number_of_negative_words
            all_social_features.append(find_in_dic("dic/first_pronoun.txt", word_seg(text)))  # number_of_first_pronoun
            all_social_features.append(find_in_dic("dic/second_pronoun.txt", word_seg(text)))  # number_of_second_pronoun
            all_social_features.append(find_in_dic("dic/third_pronoun.txt", word_seg(text)))  # number_of_third_pronoun
            all_social_features.append(text.count("https://")+text.count("http://"))  # number_of_url
            all_social_features.append(text.count('@'))  # number_of_at
            all_social_features.append(text.count('#'))  # number_of_num
            num1, num2, num3 = get_ner(text)
            all_social_features.append(num1)  # number_of_people
            all_social_features.append(num2)  # number_of_location
            all_social_features.append(num3)  # number_of_organization
            all_social_features.append(senti_score.sentiment_score(senti_score.sentiment_score_list(text)))  # Sentiment_score


            ret.append(all_social_features)
    return ret


def get_social_features_truth(filename):
    """
    类似谣言微博的获取social特征
    """
    ret = []
    it = 1
    with open(filename, 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            if it % 1000 == 0:
                print(it)
                # break
            it = it + 1
            # 循环每一个微博
            all_social_features = []
            line = line.strip()
            if not len(line):
                continue
            # print(line)
            load_dict = json.loads(line)
            text = load_dict['content']
            # print(text)
            all_social_features.append(text.count('!')+text.count('！'))  # number_of_exclamation_mark
            all_social_features.append(text.count('?') + text.count('？'))  # number_of_question_mark
            all_social_features.append(len(word_seg(text)))  # number_of_words
            all_social_features.append(len(text))  # number_of_characters
            all_social_features.append(find_in_dic("dic/emotion_dic/positive.txt", word_seg(text)))  # number_of_positive_words
            all_social_features.append(find_in_dic("dic/emotion_dic/negative.txt", word_seg(text)))  # number_of_negative_words
            all_social_features.append(find_in_dic("dic/first_pronoun.txt", word_seg(text)))  # number_of_first_pronoun
            all_social_features.append(find_in_dic("dic/second_pronoun.txt", word_seg(text)))  # number_of_second_pronoun
            all_social_features.append(find_in_dic("dic/third_pronoun.txt", word_seg(text)))  # number_of_third_pronoun
            all_social_features.append(text.count("https://")+text.count("http://"))  # number_of_url
            all_social_features.append(text.count('@'))  # number_of_at
            all_social_features.append(text.count('#'))  # number_of_num
            num1, num2, num3 = get_ner(text)
            all_social_features.append(num1)  # number_of_people
            all_social_features.append(num2)  # number_of_location
            all_social_features.append(num3)  # number_of_organization
            all_social_features.append(senti_score.sentiment_score(senti_score.sentiment_score_list(text)))  # Sentiment_score
            # print(all_social_features)
            ret.append(all_social_features)
    return ret


def get_train_data(rumor_filename, truth_filename):
    """
    得到数据集每个微博的social features
    :param rumor_filename:
    :param truth_filename:
    :return:
    """
    all_social_features = get_social_features_rumor(rumor_filename)
    num1 = len(all_social_features)
    all_social_features = all_social_features + get_social_features_truth(truth_filename)
    num2 = len(all_social_features) - num1

    print(all_social_features)
    print(len(all_social_features))
    print("提取了feature1，%d 个rumor %d个truth" % (num1, num2))

    print("转换为训练数据")
    data_x = []
    for feature in all_social_features:  # 对每个微博
        data_x.append(numpy.array(feature))
    data_x = numpy.array(data_x)
    X = data_x.reshape(num1 + num2, social_dim)
    # print(X.shape)

    # 准备标签数据
    print("准备标签数据")
    lables = [1 for i in range(0, num1)] + [0 for i in range(0, num2)]
    y = numpy.array(lables)
    y = np_utils.to_categorical(y)
    print("标签数据准备完毕")
    return X, y

if __name__ == '__main__':
    """
    提取social feature
    提取user feature
    提取other feature
    """
    train_X_dir = "npy/social_features/train_social_X.npy"
    train_y_dir = "npy/social_features/train_social_y.npy"
    val_X_dir = "npy/social_features/val_social_X.npy"
    val_y_dir = "npy/social_features/val_social_y.npy"

    flag = os.path.isfile(train_X_dir) & os.path.isfile(train_y_dir) & os.path.isfile(val_X_dir) & os.path.isfile(
        val_y_dir)
    if flag:
        print("文件已存在。")
    else:
        train_X, train_y = get_train_data("data_set/train_rumor.json", "data_set/train_truth.json")  # 训练集合
        val_X, val_y = get_train_data("data_set/test_rumor.json", "data_set/test_truth.json")  # 验证集
        numpy.save(train_X_dir, train_X)
        numpy.save(train_y_dir, train_y)
        numpy.save(val_X_dir, val_X)
        numpy.save(val_y_dir, val_y)
        print("npy数据已保存")

