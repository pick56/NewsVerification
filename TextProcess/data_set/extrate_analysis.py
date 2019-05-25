import json
import os.path

filename = "train_truth.json"
with open(filename, 'r', encoding='utf-8') as load_f:
    num = 0
    it = 1
    for line in load_f.readlines():
        # if it > 4188:
        #     break
        it = it+1
        line = line.strip()
        if not len(line):
            num = num +1
            continue
        # print(line)
        load_dict = json.loads(line)
        # print(load_dict)
        # print(load_dict['reportedWeibo'])
        try:
            # text = load_dict['reportedWeibo']['weiboContent']
            text = load_dict['content']
            # text = text.strip()
            text = text.replace('\n', '')
            text = text.replace('\t', '')
            # text = text.replace('\r', '')
            print(text)
        except:
            num = num + 1
    print(num)

# rumor_json = open(rumor_path, 'r', encoding='utf-8')
# for line in rumor_json.readlines():
#     res = json.loads(line)
#     rumor_json_list.append(res)
# for res_rumor in rumor_json_list:
#     content= res_rumor['reportedWeibo']['weiboContent']
#     res = content.replace('\n','')
#     res2 = res.replace('\t','')
#     if res2 =='':
#         res2 = 'None'
#     rumor_content.write(res2+'\t'+'0'+'\n')

