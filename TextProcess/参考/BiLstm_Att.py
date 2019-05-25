from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, Dropout, LSTM, add, Merge, Lambda, Input, merge, Flatten, Bidirectional, GlobalAveragePooling1D, multiply,concatenate
from keras.layers.core import Reshape,RepeatVector,Permute,Lambda
from keras.optimizers import SGD,Adam
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
import numpy as np
from attention_layer import AttentionLayer
from sklearn import metrics
import h5py
from keras.callbacks import EarlyStopping


word_len = 70
word_embedding = 32
topic_len=10
hidden_num=32
batch_size = 64



# train_sample_q = np.load(r'G:\data2\train_qsample.npy')
# train_sample_a = np.load(r'G:\data2\train_asample.npy')
# train_sample_t = np.load(r'G:\data2\train_tsample.npy')
# train_label = np.load(r'G:\data2\train_label.npy')
# test_sample_q = np.load(r'G:\data2\test_qsample.npy')
# test_sample_a = np.load(r'G:\data2\test_asample.npy')
# test_sample_t = np.load(r'G:\data2\test_tsample.npy')
# test_label = np.load(r'G:\data2\test_label.npy')

input_shape = (word_len, word_embedding)
input_shape1 = (topic_len, word_embedding)

q_input = Input(input_shape, name='q_input')
# shape(None,140,32)
q_lstm = Bidirectional(LSTM(hidden_num, activation='relu', return_sequences=True), merge_mode = 'concat')(q_input)
# shape(None,140,32)
qatt1_out = AttentionLayer()(q_lstm)
# shape1(None,140,1)
qatt1_rep = Lambda(lambda x : K.repeat_elements(x, hidden_num * 2, axis=2))(qatt1_out)
# shape1(None,140,32)
qmerge_out = multiply([qatt1_rep, q_lstm])
# shape(None,140,32)*shape1(None,140,32) 逐个元素相乘
qsum_out = Lambda(lambda x: K.sum(x, axis=1))(qmerge_out)
# shape(None,32)
# sum_out1=Dense(100,kernel_regularizer=l2(0.01),activation='relu')(sum_out)


a_input = Input(input_shape, name='a_input')
# shape(None,140,32)
a_lstm = Bidirectional(LSTM(hidden_num, activation='relu', return_sequences=True), merge_mode = 'concat')(a_input)
# shape(None,140,32)
aatt1_out = AttentionLayer()(a_lstm)


# shape1(None,140,1)
aatt1_rep = Lambda(lambda x : K.repeat_elements(x, hidden_num * 2, axis=2))(aatt1_out)
# shape1(None,140,32)
amerge_out = multiply([aatt1_rep, a_lstm])
# shape(None,140,32)*shape1(None,140,32) 逐个元素相乘
asum_out = Lambda(lambda x: K.sum(x, axis=1))(amerge_out)

merged = concatenate([qsum_out,asum_out])

t_input = Input(input_shape1)
t_average = GlobalAveragePooling1D()(t_input)
net = Dense(128,kernel_regularizer=l2(0.01),activation='relu')(t_average)
merged1=add([merged,net])

anonymous_out = Dense(2, activation = 'softmax', W_regularizer = l2(0.01))(merged1)

model = Model(inputs = [q_input,a_input,t_input], outputs= anonymous_out)
adam = Adam(lr = 0.0001)
early_stop = EarlyStopping(monitor='val_loss',patience=0)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

# model.fit([train_sample_q, train_sample_a,train_sample_t],train_label, epochs = 1, batch_size = batch_size, validation_data = ([test_sample_q, test_sample_a,test_sample_t],test_label), callbacks=[early_stop])
print(model.summary())
#label_pred = model.predict([sample_qtrain,sample_atrain])
# label_pred = model.predict([test_sample_q,test_sample_a])
# model.save(r'G:\data2\lstmatt_model_70_32_n.h5')