# coding=utf-8

import numpy
import time
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Bidirectional, LSTM, Input, Dropout, RepeatVector, Concatenate, GlobalAveragePooling1D, multiply
from keras.layers import Lambda
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from attention_layer import AttentionLayer


word_size = 100           # 每个微博取多少个词
word_dim = 64             # 词向量的维度
social_features_dim = 16  # social feature维度
nb_epoch = 1000           # epoch训练次数
batch_size = 100          # 批大小
input_dim = word_dim



def get_test_result(model_name, val_X, val_y):
    model_load = load_model(model_name, custom_objects={'AttentionLayer': AttentionLayer})
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

    input1 = Input(shape=(100, 64), name='text_input')
    bi = Bidirectional(LSTM(64, return_sequences=True))(input1)
    # shape1(None,100,1)
    aatt1_out = AttentionLayer()(bi)
    # print(bi)
    aatt1_rep = Lambda(lambda x: K.repeat_elements(x, 64 * 2, axis=2))(aatt1_out)
    # shape1(None,100,32)
    # print(aatt1_rep)
    # print(bi)
    amerge_out = multiply([aatt1_rep, bi])
    # shape(None,140,32)*shape1(None,140,32) 逐个元素相乘
    asum_out = Lambda(lambda x: K.sum(x, axis=1))(amerge_out)
    ans = Dense(y.shape[1], activation='softmax')(asum_out)
    model = Model(inputs=input1, output=ans)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "model_save/news__att_lstm_model.h5"
    check_point = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    model.fit(X, y, nb_epoch=nb_epoch, validation_split=0.1, batch_size=batch_size, verbose=1,callbacks=[early_stopping, check_point])
    time_end = time.time()
    print("完成模型训练。用时：%f s" % (time_end - time_start))
    print("模型已经保存为"+filepath)

    result = get_test_result(filepath, val_X, val_y)
    # print(result)
    return result


from langconv import *
import jieba


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


def get_weigth(model_name, val_X, val_y):
    from gensim.models import word2vec
    w2v_model = word2vec.Word2Vec.load(r'news_dim64_ep30.model')
    text = "@新京报: #女子警所被扒裤子#1月8日下午6时，实名认证的上海站地区治安派出所官方微博辟谣，澄清网传“女子在警所被扒裤子”不实，称系当事女子自己乱蹬致裤子脱落，后被该女子丈夫拍摄裸体下身照片。                                            "
    text = process(text)
    # print(text)
    text_seg = word_seg(text)
    st = []
    it = 1
    for word in text_seg:
        if it > 100:
            break
        it = it + 1
        try:
            vec = w2v_model.wv[word].tolist()
        except:
            vec = [0 for x in range(0, 64)]
        st.append(vec)
    for i in range(len(st), 100):
        st.append([0 for x in range(0, 64)])

    newst = []
    for temp in st:
        newst.append(numpy.array(temp))
    st = numpy.array(newst)
    batchs = []
    batchs.append(st)
    batchs = numpy.array(batchs)


    load = load_model(model_name, custom_objects={'AttentionLayer': AttentionLayer})
    print(load.summary())
    model = Model(inputs=load.input, outputs=load.get_layer('attention_layer_1').output)
    prediction = model.predict(batchs, verbose=0)  # 重新使用数据集
    print(len(prediction))
    # print(prediction[0])
    for i in range(0, len(text_seg)):
        print(text_seg[i])
        print(prediction[0][i][0]*100)
    return prediction

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
    print("从npy文件中加载数据")
    train_X = numpy.load(train_X_dir)
    train_y = numpy.load(train_y_dir)
    val_X = numpy.load(val_X_dir)
    val_y = numpy.load(val_y_dir)

    # result = get_test_result("model_save/news__att_lstm_model.h5", val_X, val_y)
    # print(result)

    # get_weigth("model_save/news__att_lstm_model.h5", val_X, val_y)
    # ssss = 1/ 0

    # 运行模型1:单纯bilstm+attention
    accuracy, precision, recall, f1 = model1(train_X, train_y, val_X, val_y)
    print("accuracy=%.2f%%" % (accuracy * 100))
    print("precision1=%.2f%% recall1=%.2f%% f11=%.2f%% " % (precision[0] * 100, recall[0] * 100, f1[0] * 100))
    print("precision2=%.2f%% recall2=%.2f%% f12=%.2f%% " % (precision[1] * 100, recall[1] * 100, f1[1] * 100))

