from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model
import os
import numpy
import json

if __name__ == '__main__':
    print("提取特征")
    load_model = load_model(r'news_model_text.model.h5')
    print(load_model.summary())
    model = Model(inputs=load_model.input, outputs=load_model.layers[3].output)
    print("从npy文件中加载数据")
    train_X = numpy.load("npy/train_X.npy")
    train_y = numpy.load("npy/train_y.npy")
    val_X = numpy.load("npy/val_X.npy")
    val_y = numpy.load("npy/val_y.npy")
    prediction = model.predict(train_X, verbose=0)  # 重新使用数据集

    print(len(prediction))
    numpy.save("npy/prediction_train.npy", prediction)

    prediction = model.predict(val_X, verbose=0)  # 重新使用数据集

    print(len(prediction))
    numpy.save("npy/prediction_test.npy", prediction)
