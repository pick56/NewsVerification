# 文本处理流程

## 1，训练word2vector
* 运行脚本train_word2vector
* 脚本会使用smallrumor.json和smalltruth.json文件训练词向量模型
* 中间文件news_seg_word_data为提取并且分词后的语料文件
* 词向量模型会保存为news_dim64_ep30.model文件供后续使用
* senti_score文件为计算情感得分脚本
* dic词典包含所有需要的词典
* npy文件夹包含，训练中保存的部分numpy数组数据
* extract_numpy_array脚本用来提取模型某一层输出
## 2，处理数据集
### 2.1，textfeature
* 每个微博截取前100个词汇，（不足100补0）
* 每个微博就变成了100个32维度的向量，用于之后再lstm中训练
* 未进行停用词的处理
### 2.2，socialfeature
* 16项socialfeature，主要做法通过词典，统计，词性分析的做法来做
* 【此处介绍每种特征的大致做法】
## 3，循环神经网络lstm
* 使用双向lstm
* 100个时间步，32维，每个时间步的输出，然后求平均，之后得到vec作为这个微博的表示

# 图片处理流程

## 1，采用预训练的resnet，不进行微调直接使用
## 2，同属一篇微博的图片进行了特征平均变成一个单一的向量
## 3，数据集只使用了同时具有图片和文本的数据

<font color=red>我是红色</font>
<font color=#008000>我是绿色</font>