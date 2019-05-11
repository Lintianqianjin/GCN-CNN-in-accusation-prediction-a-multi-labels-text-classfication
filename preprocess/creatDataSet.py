# 0 盗窃
# 2 故意伤害
# 3 诈骗
# 8 开设赌场
# 12 危险驾驶
# 16 抢劫
# 11 容留他人吸毒
# 27 信用卡诈骗

# 一个五千条

import json


def small_data():
    small_data_set = open(r'../data/小数据集.json', 'w', encoding='utf-8')
    big = open(r'../data/400词ID-罪名ID-法条ID.json', 'r', encoding='utf-8')
    accusation = {0: 0, 2: 0, 3: 0, 8: 0, 12: 0, 16: 0, 11: 0, 27: 0}
    for index, line in enumerate(big):
        print(index)
        cur = json.loads(line.strip())

        def is_in(acc):
            return acc in list(accusation.keys())

        acc_filter = list(filter(is_in, cur['accusation']))

        # 这里决定是否单分类#
        if len(acc_filter) == 1:
            for acc in acc_filter:
                if accusation[acc] == 5000:
                    break
            else:
                small_data_set.write(line)
                for acc in acc_filter:
                    accusation[acc] += 1
        # 判断是否都到了5000
        if min(accusation.values()) == 5000:
            break


def gcn_multi_labels_small_dataset():
    small_data_set = open(r'../data/GCN_dataset_smaller.json', 'w', encoding='utf-8')
    big = open(r'../data/400词ID-罪名ID-法条ID.json', 'r', encoding='utf-8')
    accusation = {0: 0, 2: 0, 3: 0, 12: 0, 16: 0}

    def is_in(acc):
        return acc in list(accusation.keys())

    with small_data_set as f:
        for index, line in enumerate(big):
            print(index)
            cur = json.loads(line.strip())

            ###筛选案由，为目标案由 开始###


            acc_filter = list(filter(is_in, cur['accusation']))
            ###筛选案由，为目标案由 结束###

            # 只有该条记录涉及到的罪名都大于等于200次了才不记录
            # if判断确认不涉及其它罪名

            if len(acc_filter) == len(cur['accusation']):
                for acc in acc_filter:
                    if accusation[acc] < 100:
                        f.write(line)
                        for acc in acc_filter:
                            accusation[acc] += 1
                        break

            # 判断是否都到了200
            if min(accusation.values()) == 100:
                break


if __name__ == '__main__':
    gcn_multi_labels_small_dataset()

    multi_times = 0
    for line in open(r'../data/GCN_dataset_smaller.json', 'r', encoding='utf-8'):
        cur = json.loads(line.strip())
        if len(cur['accusation']) > 1:
            multi_times += 1
            print(cur['accusation'])
    print(multi_times)