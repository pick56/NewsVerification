# 文本处理流程

## 1，训练word2vector
* 运行脚本train_word2vector
* 脚本会使用smallrumor.json和smalltruth.json文件训练词向量模型
* 中间文件news_seg_word_data为提取并且分词后的语料文件
* 词向量模型会保存为news_dim32_ep30.model文件供后续使用
## 2，处理数据集，变成易于处理的中间文件，