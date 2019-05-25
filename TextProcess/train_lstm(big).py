# coding=utf-8
import jieba
import jieba.posseg as pseg
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
import sys
from gensim.models import word2vec

import numpy
import time
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.model_selection import KFold

import senti_score

word_size = 100           # 每个微博取多少个词
word_dim = 250            # 词向量的维度
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


def count_words(str, words):
    ret = 0
    for word in words:
        ret += str.count(word)
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
        print(word, flag)
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
            all_social_features.append(senti_score.sentiment_score(text))  # Sentiment_score
            # print(all_social_features)
            ret.append(all_social_features)
    return ret


def get_social_features_truth(filename):
    """
    类似谣言微博的获取social特征
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
            all_social_features.append(senti_score.sentiment_score(text))  # Sentiment_score
            # print(all_social_features)
            ret.append(all_social_features)
    return ret


def get_text_features_rumor(filename, num, dim):
    w2v_model = word2vec.Word2Vec.load(r'wiki-corpus/wiki_corpus.model')

    ret = []
    with open(filename, 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            # 循环每一个微博
            line = line.strip()
            if not len(line):
                continue
            # print(line)
            load_dict = json.loads(line)
            text = load_dict['reportedWeibo']['weiboContent']
            text_seg = word_seg(text)
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


def get_text_features_truth(filename, num, dim):
    w2v_model = word2vec.Word2Vec.load(r'wiki-corpus/wiki_corpus.model')

    ret = []
    with open(filename, 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            # 循环每一个微博
            line = line.strip()
            if not len(line):
                continue
            # print(line)
            load_dict = json.loads(line)
            text = load_dict['content']
            text_seg = word_seg(text)
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


def get_scores(model_file, X, y):
    """
    对给定的模型和数据，获取分数
    :param model:
    :param X: datas
    :param y: lables
    :return:
    """
    model_load = load_model(model_file)

    # 输出模型精度，这个keras自带的函数只能返回误差和精确度
    scores = model_load.evaluate(X, y, verbose=0)
    print("Model Accuracy: %.2f%%" % (scores[1] * 100))

    # 重新在数据集上预测，获得更多指标
    prediction = model_load.predict(X, verbose=0)  # 重新使用数据集
    # print(prediction)
    y_pred = []
    for i in range(0, len(prediction)):
        index = numpy.argmax(prediction[i])
        temp = [0, 0]
        temp[index] = 1
        y_pred.append(numpy.array(temp))
    y_pred = numpy.array(y_pred)
    # print(y_pred)
    # print(y)

    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    # print("precision=%.2f%% recall=%.2f%% accuracy=%.2f%% f1=%.2f%% " % (precision, recall, accuracy, f1))
    return precision, recall, accuracy, f1


def get_test_result(model_name, val_X, val_y):
    model_load = load_model(model_name)
    # 输出模型精度，这个keras自带的函数只能返回误差和精确度
    scores = model_load.evaluate(val_X, val_y, verbose=0)
    # print("Model Accuracy: %.2f%%" % (scores[1] * 100))

    # 重新在数据集上预测，获得更多指标
    prediction = model_load.predict(val_X, verbose=0)  # 重新使用数据集
    # print(prediction)
    y_pred = []
    for i in range(0, len(prediction)):
        index = numpy.argmax(prediction[i])
        temp = [0, 0]
        temp[index] = 1
        y_pred.append(numpy.array(temp))
    y_pred = numpy.array(y_pred)
    # print(y_pred)
    # print(y)

    precision = precision_score(val_y, y_pred, average='macro')
    recall = recall_score(val_y, y_pred, average='macro')
    accuracy = accuracy_score(val_y, y_pred)
    f1 = f1_score(val_y, y_pred, average='macro')
    print("precision=%.2f%% recall=%.2f%% accuracy=%.2f%% f1=%.2f%% " % (precision, recall, accuracy, f1))
    return precision, recall, accuracy, f1


def get_train_data(rumor_filename, truth_filename):
    # 得到数据集每微博的text features
    all_text_features = get_text_features_rumor(rumor_filename, word_size, word_dim)
    num1 = len(all_text_features)
    all_text_features = all_text_features + get_text_features_truth(truth_filename, word_size, word_dim)
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


def model1(X, y, val_X, val_y):
    """
    单纯lstm+最后一个时间步输出
    :param X:
    :param y:
    :param val_X:
    :param val_y:
    :return:
    """
    print("开始训练模型1...")

    time_start = time.time()

    # kf = KFold(n_splits=5, random_state=0)
    # for split_X, split_y in kf.split(X, y):  # 五折交叉验证,太费时间
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))  # 这种情况下，lstm返回的就只是最后一个时间步的结果
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("模型结构：")
    print(model.summary())

    filepath = "model_save/news_lstm_model1-{epoch:02d}-{val_acc:.4f}.h5"
    check_point = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    model.fit(X, y, nb_epoch=nb_epoch, validation_split=0.1, batch_size=batch_size, verbose=1, callbacks=[early_stopping,check_point])

    time_end = time.time()
    print("完成模型训练。用时：%f s" % (time_end - time_start))
    print("模型已经保存为"+filepath)

    result = get_test_result(filepath, val_X, val_y)
    # print(result)
    return result


def model2(X, y):
    """
    模型将每个时间步的结果平均
    :param X:
    :param y:
    :return:
    """
    print("开始训练模型2...")
    time_start = time.time()

    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))  #

    from keras.layers import AveragePooling1D, Reshape
    # 平均池化，将100个时间步的结果平均
    model.add(AveragePooling1D(pool_size=100, strides=None, padding='valid', data_format='channels_last'))
    model.add(Reshape((32,)))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 输入的数据 X[一批大小,时间步,输入维度] 25 1 1
    print("模型结构：")
    print(model.summary())
    model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1)  # 训练模型迭代轮次。一个轮次是在整个x或y上的一轮迭代
    time_end = time.time()
    print("完成模型训练。用时：%f s" % (time_end - time_start))

    # 输出模型精度
    scores = model.evaluate(X, y, verbose=0)
    print("Model Accuracy: %.2f%%" % (scores[1] * 100))

    # 重新在数据集上预测，获得更多指标
    prediction = model.predict(X, verbose=0)  # 重新使用数据集
    # print(prediction)
    y_pred = []
    for i in range(0, len(prediction)):
        index = numpy.argmax(prediction[i])
        temp = [0, 0]
        temp[index] = 1
        y_pred.append(numpy.array(temp))
    y_pred = numpy.array(y_pred)
    # print(y_pred)
    # print(y)

    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    # print("precision=%.2f%% recall=%.2f%% accuracy=%.2f%% f1=%.2f%% " % (precision, recall, accuracy, f1))
    return precision, recall, accuracy, f1


if __name__ == '__main__':
    """
    训练lstm
    1，注意文件名,使用不同dataset时，需要更改filename的名字
    2，数据集文件不可变
    3，可以通过修改参数word_size提取每个微博词的个数
    """

    import os.path

    train_X_dir = "npy/train2_text_X.npy"
    train_y_dir = "npy/train2_text_y.npy"
    val_X_dir = "npy/val2_text_X.npy"
    val_y_dir = "npy/val2_text_y.npy"

    flag = os.path.isfile(train_X_dir) & os.path.isfile(train_y_dir) & os.path.isfile(val_X_dir) & os.path.isfile(
        val_y_dir)
    if flag:
        print("从npy文件中加载数据")
        train_X = numpy.load(train_X_dir)
        train_y = numpy.load(train_y_dir)
        val_X = numpy.load(val_X_dir)
        val_y = numpy.load(val_y_dir)
    else:
        # 训练集合
        train_X, train_y = get_train_data("data_set/train_rumor.json", "data_set/train_truth.json")
        # 验证集
        val_X, val_y = get_train_data("data_set/test_rumor.json", "data_set/test_truth.json")

        numpy.save(train_X_dir, train_X)
        numpy.save(train_y_dir, train_y)
        numpy.save(val_X_dir, val_X)
        numpy.save(val_y_dir, val_y)
        print("训练npy数据已保存")

    # 运行模型1:单纯lstm+最后一个时间步输出
    precision, recall, accuracy, f1 = model1(train_X, train_y, val_X, val_y)
    print("precision=%.2f%% recall=%.2f%% accuracy=%.2f%% f1=%.2f%% " % (precision, recall, accuracy, f1))

    # 运行模型2:单纯lstm+平均时间步输出
    # precision, recall, accuracy, f1 = model2(train_X, train_y, val_X, val_y)
    # print("precision=%.2f%% recall=%.2f%% accuracy=%.2f%% f1=%.2f%% " % (precision, recall, accuracy, f1))
