# coding=utf-8
import jieba
import json
from gensim.models import word2vec
from langconv import *
import numpy

word_size = 100           # 每个微博取多少个词
word_dim = 64             # 词向量的维度
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


def process(str):
    replace_terms = [' ', '...', '展开全文c', '谣言', '辟谣', 'quot', 'nbsp']
    for term in replace_terms:
        str = str.replace(term, ' ')
        content1 = str.strip()
        content1 = Converter('zh-hans').convert(content1)
    return content1


def get_text_rumor(filename):
    ret = []
    with open(filename, 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            # 循环每一个微博
            line = line.strip()
            if not len(line):
                continue
            # print(line)
            load_dict = json.loads(line)
            text = load_dict['reportedWeibo']['weiboContent'].strip().replace("\n", "").replace("\r", "")
            # text = process(text)
            # print(text)
            # text_seg = word_seg(text)
            # write_file(text_seg)
            # ret.append(text_seg)
            ret.append(text)
    return ret


def get_text_truth(filename):
    ret = []

    with open(filename, 'r', encoding='utf-8') as load_f:
        # it = 1
        for line in load_f.readlines():
            # it = it + 1
            # 循环每一个微博
            line = line.strip()
            if not len(line):
                # print("some thing worng")
                # print(it)
                continue
            # print(line)
            load_dict = json.loads(line)
            # if it == 705:
            #     print(load_dict)
            text = load_dict['content'].strip().replace("\n", "").replace("\r", "")
            # if it == 705:
            #     print(text)
            # text = process(text)
            # print(text)
            # text_seg = word_seg(text)
            # write_file(text_seg)
            # ret.append(text_seg)
            ret.append(text)
    return ret


def save_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for temp in data:
            f.write(temp+'\n')
    return


if __name__ == '__main__':
    """
    从json中提取文本
    """

    train_rumor_text = get_text_rumor("data_set/train_rumor.json")
    save_file("data_set/train_rumor_text.txt", train_rumor_text)

    train_truth_text = get_text_truth("data_set/train_truth.json")
    save_file("data_set/train_truth_text.txt", train_truth_text)

    test_rumor_text = get_text_rumor("data_set/test_rumor.json")
    save_file("data_set/test_rumor_text.txt", test_rumor_text)

    test_truth_text = get_text_truth("data_set/test_truth.json")
    save_file("data_set/test_truth_text.txt", test_truth_text)

