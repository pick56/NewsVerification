from gensim.models import word2vec
vector_size = 32
window_size = 5
min_count = 10
negative_size = 5
train_epoch = 30
worker_count = 30
sentences=word2vec.Text8Corpus(u'G:\data3\zhihu_all_seg_new.txt')
model=word2vec.Word2Vec(sentences, size = vector_size, window = window_size, min_count = min_count, workers = worker_count, negative = negative_size, iter = train_epoch)
model.save(r'G:\data3\soft_numpy\zhihu_dim32_ep30_model')