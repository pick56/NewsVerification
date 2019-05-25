# coding=utf-8

import numpy
from keras.models import Sequential, load_model, Model
from attention_layer import AttentionLayer
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


def get_weigth(model_name, text):
    from gensim.models import word2vec
    w2v_model = word2vec.Word2Vec.load(r'news_dim64_ep30.model')

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
    # print(load.summary())
    model = Model(inputs=load.input, outputs=load.get_layer('attention_layer_1').output)
    prediction = model.predict(batchs, verbose=0)  # 重新使用数据集
    # print(prediction[0])
    # 根据排名上色：
    # 1-20名FF0000
    # 21-40名FF4444
    # 41-60名FF8888
    # 61-80名FFEEEE
    # 81-100名FFFFFF
    # <font style="background:#FFCCCC;">测试</font>
    pre = []
    for i in range(0, 100):
        pre.append(prediction[0][i][0])
    # print(pre)
    indexs = numpy.argsort(pre)
    # print(indexs)
    # print(text_seg)
    new_seg = text_seg
    for i in range(len(text_seg), 100):
        new_seg.append("_")
    # print(len(new_seg))

    it = 0
    for index in indexs:
        if it < 20:
            color = "#FFFFFF"
        elif it < 40:
            color = "#FFCCCC"
        elif it < 60:
            color = "#FFAAAA"
        elif it < 80:
            color = "#FF6666"
        elif it < 100:
            color = "#FF0000"
        new_seg[index] = "<font style='background:"+color+";'>"+new_seg[index]+"</font>"
        # if index < len(text_seg):
        #     print("<font style='background:%s;'>%s</font>" % (color, text_seg[index]))
        # else:
        #     print("<font style='background:%s;'>%s</font>" % (color, " "))
        it = it + 1
    for i in range(0, 100):
        print(new_seg[i])
    # for i in range(0, len(text_seg)):
    #     print(text_seg[i])
    #     print(prediction[0][i][0]*100)
    print("</br></br>")
    return prediction

if __name__ == '__main__':
    """
    获取attention层的输出权重，生成html可以查看权重分布
    """
    import os.path
    text = "@新京报: #女子警所被扒裤子#1月8日下午6时，实名认证的上海站地区治安派出所官方微博辟谣，澄清网传“女子在警所被扒裤子”不实，称系当事女子自己乱蹬致裤子脱落，后被该女子丈夫拍摄裸体下身照片。                                            "

    get_weigth("model_save/news__att_lstm_model.h5", text)
    text = "【注意兴义各中小学周边小卖部“牙签弩”】此物俗称“牙签弩”，是最近中小学学生中比较常见的玩具，其威力巨大，竟能射入石膏板。于是在朋友圈以微薄之力呼吁我们的学生家长管好自己的孩子，牙签弩的生产者、营销者、消费者、相关政府职能监管部门，能够予以足够重视！射到孩子的身上，尤其是脸上，一时的伤，一世的殇【试想一下万一射到孩子的眼睛中】。家里有上学孩子的家长，禁止买来玩，【禁止销售牙签弩】大家动动手，为了孩子请转发！！！！"

    get_weigth("model_save/news__att_lstm_model.h5", text)
    text = "#笔记人生#   【重返狼群】同事从微信中发来一个视频《重返狼群》，这个视频再现了一个真实故事，有个成都女孩在草原旅行时，无意中发现一只失去了父母的狼崽，正当狼崽饥寒交迫孤立无援行将毙命之际，成都女孩就学着狼嗥呼唤狼崽，狼崽听到狼嗥竟然从昏迷中醒来，把女孩当成了自己的母亲并且追随不 ​			...展开全文c"

    get_weigth("model_save/news__att_lstm_model.h5", text)


