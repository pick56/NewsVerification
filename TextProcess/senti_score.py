import jieba
import numpy as np

# 打开词典文件，返回列表
def open_dict(Dict, path='dic/emotion_dic/'):
    path = path + '%s.txt' % Dict
    dictionary = open(path, 'r', encoding='utf-8')
    dict = []
    for word in dictionary:
        word = word.strip('\n')
        dict.append(word)
    return dict


def judgeodd(num):
    if (num % 2) == 0:
        return 'even'
    else:
        return 'odd'


deny_word = open_dict(Dict='否定词', path=r'dic/emotion_dic/')
posdict = open_dict(Dict='positive', path=r'dic/emotion_dic/')
negdict = open_dict(Dict='negative', path=r'dic/emotion_dic/')
degree_word = open_dict(Dict='程度级别词语', path=r'dic/emotion_dic/')
mostdict = degree_word[degree_word.index('extreme')+1: degree_word.index('very')]  # 权重4，即在情感词前乘以4
verydict = degree_word[degree_word.index('very')+1: degree_word.index('more')]  # 权重3
moredict = degree_word[degree_word.index('more')+1: degree_word.index('ish')]  # 权重2
ishdict = degree_word[degree_word.index('ish')+1: degree_word.index('last')]  # 权重0.5


def sentiment_score_list(sentence):

    count1 = []
    segtmp = jieba.lcut(sentence, cut_all=False)  # 把句子进行分词，以列表的形式返回
    i = 0  # 记录扫描到的词的位置
    a = 0  # 记录情感词的位置
    poscount = 0  # 积极词的第一次分值
    poscount2 = 0  # 积极词反转后的分值
    poscount3 = 0  # 积极词的最后分值（包括叹号的分值）
    negcount = 0
    negcount2 = 0
    negcount3 = 0
    for word in segtmp:
        if word in posdict:  # 判断词语是否是情感词
            poscount += 1
            c = 0
            for w in segtmp[a:i]:  # 扫描情感词前的程度词
                if w in mostdict:
                    poscount *= 4.0
                elif w in verydict:
                    poscount *= 3.0
                elif w in moredict:
                    poscount *= 2.0
                elif w in ishdict:
                    poscount *= 0.5
                elif w in deny_word:
                    c += 1
            if judgeodd(c) == 'odd':  # 扫描情感词前的否定词数
                poscount *= -1.0
                poscount2 += poscount
                poscount = 0
                poscount3 = poscount + poscount2 + poscount3
                poscount2 = 0
            else:
                poscount3 = poscount + poscount2 + poscount3
                poscount = 0
            a = i + 1  # 情感词的位置变化

        elif word in negdict:  # 消极情感的分析，与上面一致
            negcount += 1
            d = 0
            for w in segtmp[a:i]:
                if w in mostdict:
                    negcount *= 4.0
                elif w in verydict:
                    negcount *= 3.0
                elif w in moredict:
                    negcount *= 2.0
                elif w in ishdict:
                    negcount *= 0.5
                elif w in degree_word:
                    d += 1
            if judgeodd(d) == 'odd':
                negcount *= -1.0
                negcount2 += negcount
                negcount = 0
                negcount3 = negcount + negcount2 + negcount3
                negcount2 = 0
            else:
                negcount3 = negcount + negcount2 + negcount3
                negcount = 0
            a = i + 1
        elif word == '！' or word == '!':  # 判断句子是否有感叹号
            for w2 in segtmp[::-1]:  # 扫描感叹号前的情感词，发现后权值+2，然后退出循环
                if w2 in posdict or negdict:
                    poscount3 += 2
                    negcount3 += 2
                    break
        i += 1  # 扫描词位置前移

        # 以下是防止出现负数的情况
        pos_count = 0
        neg_count = 0
        if poscount3 < 0 and negcount3 > 0:
            neg_count += negcount3 - poscount3
            pos_count = 0
        elif negcount3 < 0 and poscount3 > 0:
            pos_count = poscount3 - negcount3
            neg_count = 0
        elif poscount3 < 0 and negcount3 < 0:
            neg_count = -poscount3
            pos_count = -negcount3
        else:
            pos_count = poscount3
            neg_count = negcount3

        count1.append([pos_count, neg_count])

    return count1


def sentiment_score(senti_score_list):
    score = []
    score_array = np.array(senti_score_list)
    Pos = np.sum(score_array[:, 0])
    if Pos >= 1000:
        Pos = 1000
    Neg = np.sum(score_array[:, 1])
    if Neg >= 1000:
        Neg = 1000
    # AvgPos = np.mean(score_array[:, 0])
    # AvgPos = float('%.1f' % AvgPos)
    # AvgNeg = np.mean(score_array[:, 1])
    # AvgNeg = float('%.1f' % AvgNeg)
    # StdPos = np.std(score_array[:, 0])
    # StdPos = float('%.1f' % StdPos)
    # StdNeg = np.std(score_array[:, 1])
    # StdNeg = float('%.1f' % StdNeg)
    # score.append([Pos, Neg])
    sum = Pos + Neg
    if sum == 0:
        return 0.5
    else:
        score = Pos/sum
        result = round(score, 2)  # 保留了两位小数
        return result


if __name__ == '__main__':
    data1 = '今天上海的天气真好！我的心情非常高兴！如果去旅游的话我会非常兴奋！和你一起去旅游我会更加幸福！'
    data2 = '今天上海天气真差,非常讨厌下雨,把我冻坏了,心情太不高兴了,不高兴,我真的很生气！'
    data3 = '美国华裔科学家,祖籍江苏扬州市高邮县,生于上海,斯坦福大学物理系,电子工程系和应用物理系终身教授!'
    print(sentiment_score(sentiment_score_list(data1)))
    print(sentiment_score(sentiment_score_list(data2)))
    print(sentiment_score(sentiment_score_list(data3)))
