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
from keras.layers import Dense, Bidirectional, LSTM, Input, Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.layers import AveragePooling1D, Reshape
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

import senti_score

word_size = 100           # 每个微博取多少个词
word_dim = 64             # 词向量的维度
nb_epoch = 200            # epoch训练次数
batch_size = 100           # 批大小
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


def get_text_features_rumor(filename, num, dim):
    w2v_model = word2vec.Word2Vec.load(r'news_dim64_ep30.model')

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
    w2v_model = word2vec.Word2Vec.load(r'news_dim64_ep30.model')
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
    model_load = load_model(r'news_model_text.model.h5')
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


def get_train_data(rumor_filename, truth_filename):
    """
    只提取微博文本数据
    :param truth_filename:
    :param rumor_filename:
    :return:
    """
    # filename:small_rumor.json、small_truth.json、moderate_truth.json、moderate_rumor.json
    # rumor_filename = "data_set/moderate_rumor.json"
    # truth_filename = "data_set/moderate_truth.json"

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


def model(X, y, val_X, val_y):
    """
    训练模型，并保存为模型文件
    :param X:
    :param y:
    :param val_X:
    :param val_y:
    :return:
    """
    print("开始训练模型1...")
    time_start = time.time()
    model = Sequential()
    model.add(Dropout(0.5, input_shape=(X.shape[1], X.shape[2]), noise_shape=(1, X.shape[2])))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))  # 这种情况下，lstm返回的就只是最后一个时间步的结果

    # 平均池化，将100个时间步的结果平均
    model.add(AveragePooling1D(pool_size=100, strides=None, padding='valid', data_format='channels_last'))
    model.add(Reshape((128,)))
    model.add(Dropout(0.5))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 输入的数据 X[一批大小,时间步,输入维度] 25 1 1
    # monitor 如果有验证集合，val_acc 或 val_loss ，否则就是acc和loss
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    print(model.summary())
    # 训练模型迭代轮次。一个轮次是在整个x或y上的一轮迭代
    model.fit(X, y, nb_epoch=nb_epoch, validation_data=[val_X, val_y], batch_size=batch_size, verbose=1, callbacks=[early_stopping])
    print("模型结构：")
    print(model.summary())

    model.save(r'news_model_text.model.h5')
    time_end = time.time()
    print("完成模型训练。用时：%f s" % (time_end - time_start))
    print("模型已经保存为news_model_text.model.h5")

    result = get_test_result(val_X, val_y)
    print(result)

    return result


if __name__ == '__main__':
    """
    训练lstm
    1，注意文件名,使用不同dataset时，需要更改filename的名字
    2，数据集文件不可变
    3，可以通过修改参数word_size提取每个微博词的个数
    """

    import os.path
    flag = os.path.isfile("train_X.npy") & os.path.isfile("train_y.npy") & os.path.isfile("val_X.npy") & os.path.isfile("val_y.npy")
    if flag:
        print("从npy文件中加载数据")
        train_X = numpy.load("train_X.npy")
        train_y = numpy.load("train_y.npy")
        val_X = numpy.load("val_X.npy")
        val_y = numpy.load("val_y.npy")
    else:
        # 训练集合
        train_X, train_y = get_train_data("data_set/train_rumor.json", "data_set/train_truth.json")
        # 验证集
        val_X, val_y = get_train_data("data_set/test_rumor.json", "data_set/test_truth.json")
        train_X_dir = "train_X.npy"
        train_y_dir = "train_y.npy"
        val_X_dir = "val_X.npy"
        val_y_dir = "val_y.npy"
        numpy.save(train_X_dir, train_X)
        numpy.save(train_y_dir, train_y)
        numpy.save(val_X_dir, val_X)
        numpy.save(val_y_dir, val_y)


    # 运行模型1
    precision, recall, accuracy, f1 = model(train_X, train_y, val_X, val_y)
    print("precision=%.2f%% recall=%.2f%% accuracy=%.2f%% f1=%.2f%% " % (precision, recall, accuracy, f1))
