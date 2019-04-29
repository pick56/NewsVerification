import jieba
with open(r'G:\data3\zhihu_nonanonymous.txt', 'r', encoding='UTF-8')as f1:
    with open(r'G:\data3\zhihu_nonanonymous_seg.txt', 'a', encoding='UTF-8')as f2:

        for line in f1:

            tmp = line.strip()

            content = tmp.split('\t')

            if len(content) != 5:

                continue

            question_title = content[0].strip()

            question_detail = content[1].strip()

            answer = content[2].strip()
            toptics =content[3].strip()

            qt_seg = ' '.join(jieba.cut(question_title))

            qd_seg = ' '.join(jieba.cut(question_detail))

            a_seg = ' '.join(jieba.cut(answer))

            #f2.write(qt_seg + '\t' + qd_seg + '\t' + a_seg + '\n')

            f2.write(qt_seg + '\t' + qd_seg + '\t' + a_seg +'\t'+toptics+ '\n')

        print('done')

        # for line in f1:
        #     tt = []
        #     tmp = line.replace('\n','')
        #     t = jieba.lcut(tmp,cut_all=False)
        #     # print(t)
        #     for w in t:
        #         if w !=' ':
        #             tt.append(w)
        #     # print(tt)
        #     ttt=' '.join(tt)
        #     # print(ttt)
        #
        #     f2.write(ttt+'\n')



