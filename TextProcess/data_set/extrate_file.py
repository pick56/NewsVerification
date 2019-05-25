def read_file(filename, num):
    ret = []
    i=0
    with open(filename, 'r', encoding='utf-8') as load_f:
        for line in load_f.readlines():
            line = line.strip()
            if not len(line):
                continue
            ret.append(line)
            i = i+1
            if i >= num:
                break
    return ret


def save_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for temp in data:
            f.write(temp+'\n')
    return


if __name__ == '__main__':
    """
    从data中提取数据
    
    """
    num = 4188
    # 从最大的那个数据集中提取100条写文件
    truth = read_file('train_truth.json', num)
    # save操作会自动覆盖之前的数据
    # filename:moderate_truth.json、moderate_rumor.json、small_truth.json、small_rumor.json
    # 可以在list上使用切片操作
    save_file('train_truth1.json', truth)
    xx = 1/0

    rumor = read_file('raw_rumor.json', num)
    # save操作会自动覆盖之前的数据
    save_file('moderate_rumor.json', rumor[100:1101])
    print("数据提取完毕")
