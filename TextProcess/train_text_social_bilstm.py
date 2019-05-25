# coding=utf-8
import numpy
import time
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Bidirectional, LSTM, Input, Dropout, RepeatVector, Concatenate, GlobalAveragePooling1D, multiply
from keras.layers import Lambda
from attention_layer import AttentionLayer
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import AveragePooling1D, Reshape
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

nb_epoch = 1000            # epoch训练次数
batch_size = 100          # 批大小


def get_test_result(model_name, val_tX, val_sX, val_y):
    model_load = load_model(model_name, custom_objects={'AttentionLayer': AttentionLayer})
    # 输出模型精度，这个keras自带的函数只能返回误差和精确度
    scores = model_load.evaluate([val_tX, val_sX], val_y, verbose=0)
    print("Model Accuracy: %.2f%%" % (scores[1] * 100))

    # 测试集预测
    prediction = model_load.predict([val_tX, val_sX], verbose=0)
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


def model1(t_X, s_X, y, val_tX, val_sX, val_y):
    """
    训练模型，并保存为模型文件,验证一个全连接层的重要性
    前64维是文本的，后16维是social的
    train_text_X, train_social_X, train_y, val_text_X, val_socialt_X, val_y
    :param t_X:
    :param s_X:
    :param y:
    :param val_tX:
    :param val_sX:
    :param val_y:
    :return:
    """
    print("开始训练模型1...")
    time_start = time.time()
    input1 = Input(shape=(100, 64), name='text_input')
    input2 = Input(shape=(16, ), name='social_input')

    social_re = RepeatVector(100)(input2)
    fuse = Concatenate(axis=2)([input1, social_re])

    # fuse_dro = Dropout(0.5, input_shape=(t_X.shape[1], X.shape[2]), noise_shape=(1, X.shape[2]))(fuse)
    bi = Bidirectional(LSTM(64, return_sequences=True))(fuse)
    ave = GlobalAveragePooling1D(data_format='channels_last')(bi)
    prediction = Dense(y.shape[1], activation='softmax')(ave)

    model = Model(inputs=[input1, input2], output=prediction)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = "model_save/news_social_bilstm_model1.h5"
    check_point = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    model.fit([t_X, s_X], y, nb_epoch=nb_epoch, validation_split=0.1, batch_size=batch_size, verbose=1,callbacks=[early_stopping, check_point])

    print("模型结构：")
    print(model.summary())
    time_end = time.time()
    print("完成模型训练。用时：%f s" % (time_end - time_start))
    print("模型已经保存为"+filepath)

    result = get_test_result(filepath, val_tX, val_sX, val_y)
    return result


def model2(t_X, s_X, y, val_tX, val_sX, val_y):
    """
    过一个dense再融合
    前64维是文本的，后16维是social的
    train_text_X, train_social_X, train_y, val_text_X, val_socialt_X, val_y
    :param t_X:
    :param s_X:
    :param y:
    :param val_tX:
    :param val_sX:
    :param val_y:
    :return:
    """
    print("开始训练模型2...")
    time_start = time.time()
    input1 = Input(shape=(100, 64), name='text_input')
    input2 = Input(shape=(16,), name='social_input')

    social_fc = Dense(16, activation="sigmoid")(input2)
    social_fc_re = RepeatVector(100)(social_fc)
    fuse = Concatenate(axis=2)([input1, social_fc_re])

    # fuse_dro = Dropout(0.5, input_shape=(t_X.shape[1], X.shape[2]), noise_shape=(1, X.shape[2]))(fuse)
    bi = Bidirectional(LSTM(64, return_sequences=True))(fuse)
    ave = GlobalAveragePooling1D(data_format='channels_last')(bi)
    prediction = Dense(y.shape[1], activation='softmax')(ave)

    model = Model(inputs=[input1, input2], output=prediction)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = "model_save/news_social_bilstm_model2.h5"
    check_point = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    model.fit([t_X, s_X], y, nb_epoch=nb_epoch, validation_split=0.1, batch_size=batch_size, verbose=1,
              callbacks=[early_stopping, check_point])

    print("模型结构：")
    print(model.summary())
    time_end = time.time()
    print("完成模型训练。用时：%f s" % (time_end - time_start))
    print("模型已经保存为" + filepath)

    result = get_test_result(filepath, val_tX, val_sX, val_y)
    return result


def model3(t_X, s_X, y, val_tX, val_sX, val_y):
    """
    等text经过bilstm之后再进行融合
    前64维是文本的，后16维是social的
    train_text_X, train_social_X, train_y, val_text_X, val_socialt_X, val_y
    :param t_X:
    :param s_X:
    :param y:
    :param val_tX:
    :param val_sX:
    :param val_y:
    :return:
    """
    print("开始训练模型3...")
    time_start = time.time()
    input1 = Input(shape=(100, 64), name='text_input')
    input2 = Input(shape=(16,), name='social_input')

    bi = Bidirectional(LSTM(64, return_sequences=True))(input1)
    ave = GlobalAveragePooling1D(data_format='channels_last')(bi)
    social_fc = Dense(16, activation="sigmoid")(input2)
    fuse = Concatenate(axis=1)([ave, social_fc])
    prediction = Dense(y.shape[1], activation='softmax')(fuse)

    model = Model(inputs=[input1, input2], output=prediction)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = "model_save/news_social_bilstm_model3.h5"
    check_point = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    model.fit([t_X, s_X], y, nb_epoch=nb_epoch, validation_split=0.1, batch_size=batch_size, verbose=1,
              callbacks=[early_stopping, check_point])

    print("模型结构：")
    print(model.summary())
    time_end = time.time()
    print("完成模型训练。用时：%f s" % (time_end - time_start))
    print("模型已经保存为" + filepath)

    result = get_test_result(filepath, val_tX, val_sX, val_y)
    return result


def model4(t_X, s_X, y, val_tX, val_sX, val_y):
    """
    在模型2的基础增加attention机制
    前64维是文本的，后16维是social的
    train_text_X, train_social_X, train_y, val_text_X, val_socialt_X, val_y
    :param t_X:
    :param s_X:
    :param y:
    :param val_tX:
    :param val_sX:
    :param val_y:
    :return:
    """
    print("开始训练模型4...")
    time_start = time.time()
    input1 = Input(shape=(100, 64), name='text_input')
    input2 = Input(shape=(16,), name='social_input')

    social_fc = Dense(16, activation="sigmoid")(input2)
    social_fc_re = RepeatVector(100)(social_fc)
    fuse = Concatenate(axis=2)([input1, social_fc_re])

    # fuse_dro = Dropout(0.5, input_shape=(t_X.shape[1], X.shape[2]), noise_shape=(1, X.shape[2]))(fuse)
    bi = Bidirectional(LSTM(64, return_sequences=True))(fuse)

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
    prediction = Dense(y.shape[1], activation='softmax')(asum_out)

    model = Model(inputs=[input1, input2], output=prediction)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = "model_save/news_social_bilstm_model(attention).h5"
    check_point = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    model.fit([t_X, s_X], y, nb_epoch=nb_epoch, validation_split=0.1, batch_size=batch_size, verbose=1,
              callbacks=[early_stopping, check_point])

    print("模型结构：")
    print(model.summary())
    time_end = time.time()
    print("完成模型训练。用时：%f s" % (time_end - time_start))
    print("模型已经保存为" + filepath)

    result = get_test_result(filepath, val_tX, val_sX, val_y)
    return result


if __name__ == '__main__':
    """
    结合social feature和text feature训练bilstm
    """
    train_text_X = numpy.load("npy/train_text_X.npy")
    train_social_X = numpy.load("npy/social_features/train_social_X.npy")
    train_y = numpy.load("npy/train_text_y.npy")

    val_text_X = numpy.load("npy/val_text_X.npy")
    val_socialt_X = numpy.load("npy/social_features/val_social_X.npy")
    val_y = numpy.load("npy/val_text_y.npy")

    # get_test_result("model_save/news_social_bilstm_model(attention).h5", val_text_X, val_socialt_X, val_y)
    # sss = 1/0

    # 运行模型1: 0dense+lstm前融合
    # accuracy, precision, recall, f1 = model1(train_text_X, train_social_X, train_y, val_text_X, val_socialt_X, val_y)
    # print("accuracy=%.2f%%" % (accuracy * 100))
    # print("precision1=%.2f%% recall1=%.2f%% f11=%.2f%% " % (precision[0] * 100, recall[0] * 100, f1[0] * 100))
    # print("precision2=%.2f%% recall2=%.2f%% f12=%.2f%% " % (precision[1] * 100, recall[1] * 100, f1[1] * 100))

    # 运行模型2: 1dense+lstm前融合
    # accuracy, precision, recall, f1 = model2(train_text_X, train_social_X, train_y, val_text_X, val_socialt_X, val_y)
    # print("accuracy=%.2f%%" % (accuracy * 100))
    # print("precision1=%.2f%% recall1=%.2f%% f11=%.2f%% " % (precision[0] * 100, recall[0] * 100, f1[0] * 100))
    # print("precision2=%.2f%% recall2=%.2f%% f12=%.2f%% " % (precision[1] * 100, recall[1] * 100, f1[1] * 100))

    # 运行模型3: 1dense+lstm后融合
    accuracy, precision, recall, f1 = model3(train_text_X, train_social_X, train_y, val_text_X, val_socialt_X, val_y)
    print("accuracy=%.2f%%" % (accuracy * 100))
    print("precision1=%.2f%% recall1=%.2f%% f11=%.2f%% " % (precision[0] * 100, recall[0] * 100, f1[0] * 100))
    print("precision2=%.2f%% recall2=%.2f%% f12=%.2f%% " % (precision[1] * 100, recall[1] * 100, f1[1] * 100))

    # 运行模型4: 1dense+lstm前融合+attention
    # accuracy, precision, recall, f1 = model4(train_text_X, train_social_X, train_y, val_text_X, val_socialt_X, val_y)
    # print("accuracy=%.2f%%" % (accuracy * 100))
    # print("precision1=%.2f%% recall1=%.2f%% f11=%.2f%% " % (precision[0] * 100, recall[0] * 100, f1[0] * 100))
    # print("precision2=%.2f%% recall2=%.2f%% f12=%.2f%% " % (precision[1] * 100, recall[1] * 100, f1[1] * 100))

