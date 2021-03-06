# coding=utf-8
import jieba
import jieba.posseg as pseg
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
import sys
from gensim.models import word2vec
from langconv import *
import numpy
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, Bidirectional, LSTM, Input, Dropout, GlobalAveragePooling1D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
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

def process(str):
    replace_terms = [' ', '...', '展开全文c', '谣言', '辟谣', 'quot', 'nbsp']
    for term in replace_terms:
        str = str.replace(term, ' ')
        content1 = str.strip()
        content1 = Converter('zh-hans').convert(content1)
    return content1

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
            text = process(text)
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
            text = process(text)
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


def get_test_result(model_name, val_X, val_y):
    model_load = load_model(model_name)
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

    accuracy = accuracy_score(val_y, y_pred)
    precision = precision_score(val_y, y_pred, average=None).tolist()
    recall = recall_score(val_y, y_pred, average=None).tolist()
    f1 = f1_score(val_y, y_pred, average=None).tolist()
    print("accuracy=%.2f%%" % (accuracy * 100))
    print("precision1=%.2f%% recall1=%.2f%% f11=%.2f%% " % (precision[0] * 100, recall[0] * 100, f1[0] * 100))
    print("precision2=%.2f%% recall2=%.2f%% f12=%.2f%% " % (precision[1] * 100, recall[1] * 100, f1[1] * 100))

    return accuracy, precision, recall, f1


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

    model.add(Bidirectional(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)))
    # model.add(Dropout(0.5, input_shape=(X.shape[1], X.shape[2]), noise_shape=(1, X.shape[2])))
    # model.add(Bidirectional(LSTM(64, return_sequences=True)))

    # 平均池化，将100个时间步的结果平均
    model.add(GlobalAveragePooling1D(data_format='channels_last'))
    model.add(Reshape((128,)))
    # model.add(Dropout(0.5))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "model_save/news_bilstm_model1(dropout).h5"
    check_point = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    # 训练模型迭代轮次。一个轮次是在整个x或y上的一轮迭代
    model.fit(X, y, nb_epoch=nb_epoch, validation_split=0.1, batch_size=batch_size, verbose=1, callbacks=[early_stopping, check_point])
    print("模型结构：")
    print(model.summary())
    time_end = time.time()
    print("完成模型训练。用时：%f s" % (time_end - time_start))
    print("模型已经保存为"+filepath)

    result = get_test_result(filepath, val_X, val_y)
    # print(result)
    return result


def model_dropout(X, y, val_X, val_y):
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

    # model.add(Bidirectional(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)))
    model.add(Dropout(0.5, input_shape=(X.shape[1], X.shape[2]), noise_shape=(1, X.shape[2])))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))

    # 平均池化，将100个时间步的结果平均
    model.add(GlobalAveragePooling1D(data_format='channels_last'))
    model.add(Reshape((128,)))
    model.add(Dropout(0.5))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "model_save/news_bilstm_model1(dropout).h5"
    check_point = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    # 训练模型迭代轮次。一个轮次是在整个x或y上的一轮迭代
    model.fit(X, y, nb_epoch=nb_epoch, validation_split=0.1, batch_size=batch_size, verbose=1, callbacks=[early_stopping, check_point])
    print("模型结构：")
    print(model.summary())
    time_end = time.time()
    print("完成模型训练。用时：%f s" % (time_end - time_start))
    print("模型已经保存为"+filepath)

    result = get_test_result(filepath, val_X, val_y)
    # print(result)
    return result


if __name__ == '__main__':
    """
    训练lstm
    1，注意文件名,使用不同dataset时，需要更改filename的名字
    2，数据集文件不可变
    3，可以通过修改参数word_size提取每个微博词的个数
    """

    import os.path

    train_X_dir = "npy/train_text_X.npy"
    train_y_dir = "npy/train_text_y.npy"
    val_X_dir = "npy/val_text_X.npy"
    val_y_dir = "npy/val_text_y.npy"

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


    # 运行模型:单纯bilstm+时间步平均
    # accuracy, precision, recall, f1 = model(train_X, train_y, val_X, val_y)
    # print("accuracy=%.2f%%" % (accuracy * 100))
    # print("precision1=%.2f%% recall1=%.2f%% f11=%.2f%% " % (precision[0] * 100, recall[0] * 100, f1[0] * 100))
    # print("precision2=%.2f%% recall2=%.2f%% f12=%.2f%% " % (precision[1] * 100, recall[1] * 100, f1[1] * 100))

    # 运行模型:单纯bilstm+dropout时间步平均
    accuracy, precision, recall, f1 = model_dropout(train_X, train_y, val_X, val_y)
    print("accuracy=%.2f%%" % (accuracy * 100))
    print("precision1=%.2f%% recall1=%.2f%% f11=%.2f%% " % (precision[0] * 100, recall[0] * 100, f1[0] * 100))
    print("precision2=%.2f%% recall2=%.2f%% f12=%.2f%% " % (precision[1] * 100, recall[1] * 100, f1[1] * 100))
