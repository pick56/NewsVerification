# coding=utf-8
import jieba
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
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, accuracy_score ,recall_score, f1_score

word_size = 100           # 每个微博取多少个词
word_dim = 32             # 词向量的维度
social_features_dim = 16  # social feature维度
nb_epoch = 1000           # epoch训练次数
batch_size = 40           # 批大小
input_dim = word_dim + social_features_dim

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
            all_social_features.append(0)  # Sentiment_score
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
            all_social_features.append(0)  # Sentiment_score
            # print(all_social_features)
            ret.append(all_social_features)
    return ret


def get_text_features_rumor(filename, num, dim):
    w2v_model = word2vec.Word2Vec.load(r'news_dim32_ep30.model')

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
                st.append(w2v_model.wv[word].tolist())
            for i in range(len(st), num):
                st.append([0 for x in range(0, dim)])
            ret.append(st)
        # print(len(ret))
    return ret


def get_text_features_truth(filename, num, dim):
    w2v_model = word2vec.Word2Vec.load(r'news_dim32_ep30.model')
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
                st.append(w2v_model.wv[word].tolist())
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


def get_test_result(val_X, val_y):
    model_load = load_model(r'news_model_1.model.h5')
    # 输出模型精度，这个keras自带的函数只能返回误差和精确度
    scores = model_load.evaluate(val_X, val_y, verbose=0)
    print("Model Accuracy: %.2f%%" % (scores[1] * 100))

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


def model_1(X, y, val_X, val_y):
    print("开始训练模型1...")
    time_start = time.time()
    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))  # 这种情况下，lstm返回的就只是最后一个时间步的结果
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 输入的数据 X[一批大小,时间步,输入维度] 25 1 1
    print("模型结构：")
    print(model.summary())
    # monitor 如果有验证集合，val_acc 或 val_loss ，否则就是acc和loss
    early_stopping = EarlyStopping(monitor='loss', patience=5)

    # 训练模型迭代轮次。一个轮次是在整个x或y上的一轮迭代
    model.fit(X, y, nb_epoch=nb_epoch, validation_split=0.1, batch_size=batch_size, verbose=1, callbacks=[early_stopping])

    model.save(r'news_model_1.model.h5')
    time_end = time.time()
    print("完成模型训练。用时：%f s" % (time_end - time_start))
    print("模型已经保存为news_model_1.model.h5")

    result = get_test_result(val_X, val_y)
    print(result)

    return

    model_load = load_model(r'news_model_1.model.h5')
    sc = model_load.evaluate(X, y, verbose=0)
    print(sc)


    # 我的验证集==还没弄
    # 输出模型精度，这个keras自带的函数只能返回误差和精确度
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


def model_2(X, y):
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


def get_train_data(truth_filename, rumor_filename):
    # filename:small_rumor.json、small_truth.json、moderate_truth.json、moderate_rumor.json
    # truth_filename = "data_set/moderate_truth.json"
    # rumor_filename = "data_set/moderate_rumor.json"

    # 得到数据集每个微博的social features
    all_social_features = get_social_features_rumor(rumor_filename)
    num1 = len(all_social_features)
    all_social_features = all_social_features + get_social_features_truth(truth_filename)
    num2 = len(all_social_features) - num1

    # print(all_social_features)
    # print(len(all_social_features))
    print("提取了social features，%d 个rumor %d个truth" % (num1, num2))

    # 得到数据集每微博的text features
    all_text_features = get_text_features_rumor(rumor_filename, word_size, word_dim)
    num1 = len(all_text_features)
    all_text_features = all_text_features + get_text_features_truth(truth_filename, word_size, word_dim)
    num2 = len(all_text_features) - num1

    print("提取了text features，%d 个rumor %d个truth" % (num1, num2))

    print("转换为训练数据")
    data_x = []
    for i in range(0, num1 + num2):  # 对每个微博
        temp_x = []
        for temp in all_text_features[i]:  # 100个词
            # print(len(temp + all_social_features[i]))
            temp_x.append(numpy.array(temp + all_social_features[i]))
        data_x.append(numpy.array(temp_x))
    # print(len(datax))
    data_x = numpy.array(data_x)
    X = data_x.reshape(num1 + num2, word_size, input_dim)
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
    训练lstm
    1，注意文件名,使用不同dataset时，需要更改filename的名字
    2，数据集文件不可变
    3，可以通过修改参数word_size提取每个微博词的个数
    """
    # 训练集合
    train_X, train_y = get_train_data("data_set/moderate_truth.json", "data_set/moderate_rumor.json")
    # 验证集
    val_X, val_y = get_train_data("data_set/small_truth.json", "data_set/small_rumor.json")

    # 运行模型1
    precision, recall, accuracy, f1 = model_1(train_X, train_y, val_X, val_y)
    print("precision=%.2f%% recall=%.2f%% accuracy=%.2f%% f1=%.2f%% " % (precision, recall, accuracy, f1))

    # 运行模型2
    # precision, recall, accuracy, f1 = model_2(X, y)
    # print("precision=%.2f%% recall=%.2f%% accuracy=%.2f%% f1=%.2f%% " % (precision, recall, accuracy, f1))

'''
if __name__ == '__main__' and sys.argv[1]=='train_word2vector':

问题：
1，truth中21个属性

2，rumor17个属性，10个是微博的，7个是用户的

3，都有的共同属性14-1个，其中1个不确定

4，哪个部分是textual feature
微博内容，截断前100个字符
得到100个32维度的向量

5，social context feature
得到1个16维的向量

6，1个16维的向量全连接变成1个32维的向量
7，一个微博的textual feature和social textual feature就是71个32维的向量
8，然后用71个cell的lstm，跑一个many2many得到71个32维的向量，平均一下就是一个32维的向量
9，然后和图片融合，最基本的融合就是图片的32维和文本的32维拼接在一起，变成64维度的东西
10，然后全连接层，做一个softmax二分类
'''