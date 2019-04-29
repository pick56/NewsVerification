import numpy as np
from gensim.models import Word2Vec


# anonymous: [1, 0]
# non_anonymous: [0, 1]

word_length = 70
word_embedding = 32
topic_length=10

def gen_word_npy(txt_file, np_qsample_file,np_asample_file,np_tsample_file,np_label_file):
    w2v_model = Word2Vec.load(r'G:\data2\zhihu_dim32_ep30_model')
    qsample_np_list = []
    asample_np_list = []
    tsample_np_list = []
    label_np_list = []
    with open(txt_file,encoding='utf-8') as f:
        for line in f.readlines():
            content = line.strip().split('\t')

            if len(content) != 5:
                print('i')
                continue
            else:
                label = content[4]
                ttxt = content[3]
                question_title = content[0].strip()
                question_detail = content[1].strip()
                qtxt = question_title+' '+question_detail
                atxt = content[2]



            if label == '1':
                label_np = np.array([1.0]).reshape(1,1)
            else:
                label_np = np.array([0.0]).reshape(1,1)
            label_np_list.append(label_np)


            qtxt_seg = qtxt.split(' ')
            # pad or truncate
            if len(qtxt_seg) > word_length:
                qtxt_seg = qtxt_seg[0: word_length]
            else:
                for index in range(len(qtxt_seg), word_length):
                    qtxt_seg.append(None)

            # generate sample
            qvec_list = []
            for word in qtxt_seg:
                # vec = w2v_model[word.encode('utf-8')].reshape(1, word_embedding)
                try:
                    vec = w2v_model[word].reshape(1, word_embedding)
                    # print('1')
                except:
                    # print('error')
                    vec = np.zeros([1, word_embedding])
                qvec_list.append(vec)

            qsample_np = np.concatenate([vec for vec in qvec_list]).reshape([1, word_length, word_embedding])
            qsample_np_list.append(qsample_np)

            atxt_seg = atxt.split(' ')
            # pad or truncate
            if len(atxt_seg) > word_length:
                atxt_seg = atxt_seg[0: word_length]
            else:
                for index in range(len(atxt_seg), word_length):
                    atxt_seg.append(None)

            # generate sample
            avec_list = []
            for word in atxt_seg:
                # vec = w2v_model[word.encode('utf-8')].reshape(1, word_embedding)
                try:
                    vec = w2v_model[word].reshape(1, word_embedding)
                    # print('1')
                except:
                    # print('error')
                    vec = np.zeros([1, word_embedding])
                avec_list.append(vec)

            asample_np = np.concatenate([vec for vec in avec_list]).reshape([1, word_length, word_embedding])
            asample_np_list.append(asample_np)



            ttxt_seg = ttxt.split(' ')
            # pad or truncate
            if len(ttxt_seg) > topic_length:
                ttxt_seg = ttxt_seg[0: topic_length]
            else:
                for index in range(len(ttxt_seg), topic_length):
                    ttxt_seg.append(None)

            # generate sample
            tvec_list = []
            for word in ttxt_seg:
                # vec = w2v_model[word.encode('utf-8')].reshape(1, word_embedding)
                try:
                    vec = w2v_model[word].reshape(1, word_embedding)
                    # print('1')
                except:
                    # print('error')
                    vec = np.zeros([1, word_embedding])
                tvec_list.append(vec)

            tsample_np = np.concatenate([vec for vec in tvec_list]).reshape([1, topic_length, word_embedding])
            tsample_np_list.append(tsample_np)


    qsample = np.concatenate([vec for vec in qsample_np_list])
    asample = np.concatenate([vec for vec in asample_np_list])
    tsample = np.concatenate([vec for vec in tsample_np_list])
    label = np.concatenate([vec for vec in label_np_list])
    np.save(np_qsample_file, qsample)
    np.save(np_asample_file, asample)
    np.save(np_tsample_file, tsample)
    np.save(np_label_file,label)
if __name__ == '__main__':
    print('begin')
    # gen_word_npy(r'F:\data\zhihu_train_sample.txt', r'F:\data\final_70_10\train_qsample.npy', r'F:\data\final_70_10\train_asample.npy',r'F:\data\final_70_10\train_label.npy')

    gen_word_npy(r'G:\data3\zhihu_train_sample.txt', r'G:\data3\num\train_qsample.npy',
                 r'G:\data3\num\train_asample.npy', r'G:\data3\num\train_tsample.npy',r'G:\data3\num\train_label.npy')
    gen_word_npy(r'G:\data3\zhihu_test_sample.txt', r'G:\data3\num\test_qsample.npy',
                 r'G:\data3\num\test_asample.npy', r'G:\data3\num\test_tsample.npy',r'G:\data3\num\test_label.npy')
    print('done')