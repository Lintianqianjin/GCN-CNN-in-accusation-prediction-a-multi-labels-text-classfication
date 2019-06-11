import math
from scipy import sparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import regex as re

import json

class tools():

    def __init__(self,BaseDir = None):

        self.BaseDir = BaseDir

        if not os.path.exists(self.BaseDir):
            os.makedirs(self.BaseDir)

    def getTimeStamp(self):
        import time
        # import datetime
        T = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
        return T

    def constructLabels(self):
        accusation = {0: 0, 2: 1, 3: 2, 12: 3, 16: 4}
        dataSet = open(os.path.join(self.BaseDir,f'GCN_dataset.json'),'r',encoding='utf-8')
        labels = []
        for line in tqdm(dataSet):
            data = json.loads(line.strip())
            acc = data['accusation']
            label = np.zeros(shape=(5),dtype=int)
            for _ in acc:
                label[accusation[_]] = 1
            labels.append(label)

        # for blank_lable in range(8182):
        #     labels.append(np.zeros(shape=(5),dtype=int))

        np.save(os.path.join(self.BaseDir,'labels.npy'),arr=labels)

    def constructOnehotFeatures(self):
        filenames = os.listdir(self.BaseDir)
        for filename in filenames:
            if filename.startswith('GCN_pointsIdx_'):
                num_Nodes = int(re.search('Idx_(?P<num_Nodes>\d*)\.',filename).group('num_Nodes'))
                break
        else:
            print('please run generatePointsIndex() first')
            return None
        row = range(num_Nodes)
        col = range(num_Nodes)
        data = np.ones(shape=(1,num_Nodes),dtype=float).flatten()
        sparse_matrix = sparse.coo_matrix((data, (row, col)), shape=(num_Nodes, num_Nodes))
        sparse.save_npz(os.path.join(self.BaseDir,f'onehotFeatures.npz'),
                        sparse_matrix)

    def constructMap(self):
        # 加载词间边文件
        pmi_edge = np.loadtxt(os.path.join(self.BaseDir,f'GCN_wordsPmiEdge.txt'),
                                encoding='utf-8',delimiter=',')
        # 加载文档与词间边文件
        tfidf_edge = np.loadtxt(os.path.join(self.BaseDir,f'GCN_docWordEdge.txt'),
                                encoding='utf-8',delimiter=',')
        all_edge = np.vstack((pmi_edge,tfidf_edge))

        row = np.array(all_edge[:,0:1],dtype=int).flatten()
        col = np.array(all_edge[:,1:2],dtype=int).flatten()
        data = np.array(all_edge[:,2:],dtype=float).flatten()
        sparse_matrix = sparse.coo_matrix((data, (row, col)))
        # 保存为本地的稀疏矩阵形式 .npz
        sparse.save_npz(os.path.join(self.BaseDir,'mapStructure.npz'),sparse_matrix)

    def generateEdgesBetweenWords(self):
        # 本方法读取词语的PMI的窗口信息，生成词语之间的稀疏矩阵
        # PMI windows文件
        filenames = os.listdir(self.BaseDir)
        # 加载节点与索引对照
        # 用于存储边的时候得到坐标
        for filename in filenames:
            if filename.startswith('GCN_getPointIndex'):
                points_idx_dict = json.load(
                    open(os.path.join(self.BaseDir, filename), 'r', encoding='utf-8'))
                break
        else:
            print('please run generatePointsIndex() first')
            return None

        for filename in filenames:
            if filename.startswith('GCN_PMIWindows_'):
                PMIwindows = json.load(
                    open(os.path.join(self.BaseDir, filename), 'r', encoding='utf-8'))
                windows_counts = int(re.search('PMIWindows_(?P<num_windows>\d*)\.',filename).group('num_windows'))
                print(f"windows_counts {windows_counts}")
                break
        else:
            print('please run countPMIWindows() first')
            return None

        # 边的文件，行，列，值
        # 循环计算两单词的PMI
        pmi_edges = open(os.path.join(self.BaseDir,f'GCN_wordsPmiEdge.txt'),'w',encoding='utf-8')
        with pmi_edges as f:
            for word in tqdm(PMIwindows):
                # 该词与其他词的共现窗口情况
                cooc_with_other_words = PMIwindows[word]['cooccurence']
                # 循环计算与其他词的PMI值
                for other_word in cooc_with_other_words:
                    # 共现窗口
                    pij = cooc_with_other_words[other_word]/windows_counts
                    # 含i的窗口
                    pi = PMIwindows[word]['self']/windows_counts
                    # 含j的窗口
                    pj = PMIwindows[other_word]['self']/windows_counts
                    # 计算pmi
                    pmi_ij = math.log10(pij/(pi*pj))
                    # 如果大于0，该边写入文件
                    if pmi_ij > 0:
                        row = points_idx_dict[f'word{word}']
                        col = points_idx_dict[f'word{other_word}']
                        value = pmi_ij
                        # print([row,col,value])
                        f.write(f'{row},{col},{value}\n')

    def generateEdgesBetweenDocsWords(self):
        # 本方法读取TF和IDF文件，生成词语和文本之间的稀疏矩阵
        filenames = os.listdir(self.BaseDir)

        # 加载节点与索引对照
        # 用于存储边的时候得到坐标
        for filename in filenames:
            if filename.startswith('GCN_getPointIndex'):
                points_idx_dict = json.load(
                    open(os.path.join(self.BaseDir,filename),'r',encoding='utf-8'))
                break
        else:
            print('please run generatePointsIndex() first')
            return None

        # 加载IDF文件
        for filename in filenames:
            if filename.startswith('GCN_IDF'):
                word_IDF = json.load(
                    open(os.path.join(self.BaseDir, filename),
                         'r',
                         encoding='utf-8'))
                break
        else:
            print('please run countIDF() first')
            return None

        # 加载TF文件
        for filename in filenames:
            if filename.startswith('GCN_TF'):
                doc_TF = json.load(
                    open(os.path.join(self.BaseDir, filename),
                         'r',
                         encoding='utf-8'))
                break
        else:
            print('please run countTF() first')
            return None
        # 建立稀疏记录的dataframe
        # tfidf_edges = pd.DataFrame(columns=('行', '列', '值'))
        # 循环计算TF文档中每个单词的TF-IDF值
        with open(os.path.join(self.BaseDir,f'GCN_docWordEdge.txt'),'w',encoding='utf-8') as f:
            for doc in tqdm(doc_TF):
                for word in doc_TF[doc]:
                    # 根据节点索引生成边，写入文档与词之间边的文件
                    tfidf = doc_TF[doc][word]*word_IDF[word]
                    row = points_idx_dict[f'doc{doc}']
                    col = points_idx_dict[f'word{word}']
                    f.write(f'{row},{col},{tfidf}\n')
                    # 对称
                    f.write(f'{col},{row},{tfidf}\n')

    def generatePointsIndex(self):
        # 生成文档+单词节点对应索引的文档
        points_idx = dict()
        # 前面为文档,后面为单词
        p_idx = 0
        # 先从 tf 文件里加入doc索引
        # 即前面的节点是文档
        # 后面的节点是词语
        filenames = os.listdir(self.BaseDir)
        for filename in filenames:
            if filename.startswith('GCN_TF_'):
                docs = json.load(open(os.path.join(self.BaseDir,filename),'r',encoding='utf-8'))
                break
        else:
            print('please run countTF() first')
            return None
        # TF 文件里的key就是文档样本的索引序号
        for key in docs:
            points_idx[p_idx] = f'doc{key}'
            print(points_idx[p_idx])
            p_idx += 1

        for filename in filenames:
            if filename.startswith('GCN_IDF_'):
                # 从idf文件里 读单词ID并加入点的索引
                words = json.load(open(os.path.join(self.BaseDir,filename),'r',encoding='utf-8'))
                break
        else:
            print('please run countIDF() first')
            return None
        # words里的key就是word的序号
        for key in words:
            points_idx[p_idx] = f'word{key}'
            print(points_idx[p_idx])
            p_idx+=1

        with open(os.path.join(self.BaseDir,f'GCN_pointsIdx_{p_idx}.json'),'w',encoding='utf-8') as f:
            f.write(json.dumps(points_idx,ensure_ascii=False,indent=1))
        with open(os.path.join(self.BaseDir,f'GCN_getPointIndex.json'), 'w',
                  encoding='utf-8') as f:
            dict_ = {v: k for k, v in points_idx.items()}
            f.write(json.dumps(dict_, ensure_ascii=False, indent=1))

    def countTF(self):
        dataset = open(os.path.join(self.BaseDir,f'GCN_dataset.json'),'r',encoding='utf-8')
        tfdict = dict()
        for index,sample in enumerate(dataset):
            # 建立字典记录当前样本的各词词频
            # 需要和数据集里的样本顺序保持一致
            tfdict[index] = dict()
            words = json.loads(sample.strip())['fact']
            for word in words:
                # 0就开始填充了
                if word == 0:
                    break
                if word in tfdict[index]:
                    tfdict[index][word] += 1
                else:
                    tfdict[index][word] = 1
        with open(os.path.join(self.BaseDir, f'GCN_TF_{index+1}.json'), 'w', encoding='utf-8') as tffile:
            tffile.write(json.dumps(tfdict,ensure_ascii=False,indent=1))

    def countIDF(self):
        # 用于计算当前数据集中所有单词的IDF值
        # 数据格式为{'word':IDF}

        dataset = open(os.path.join(self.BaseDir,f'GCN_dataset.json'), 'r', encoding='utf-8')

        # 先计算所有的单词的出现的文档次数
        dfdict = dict()
        for index, sample in enumerate(dataset):
            # 建立字典记录当前样本DF值
            words = json.loads(sample.strip())['fact']
            # set 去重
            worddistinct = set(words)
            for word in worddistinct:
                if word != 0:
                    if word not in dfdict:
                        dfdict[word] = 1
                    else:
                        dfdict[word] += 1
        # 循环完后，dfdict 里记录了所有出现过的单词，以及它们的文档频率
        # 现在对所有的值除字典长度，再取倒数，再取对数即得逆文档频率
        d_ = len(dfdict)
        # 开始对每个词做IDF计算
        for w in dfdict:
            dfdict[w] = math.log10(d_/dfdict[w])

        sorted_dfdict = dict(sorted(dfdict.items(),key=lambda item:item[1]))

        with open(os.path.join(self.BaseDir,f'GCN_IDF_{d_}.json'),
                  'w', encoding='utf-8') as f:
            f.write(json.dumps(sorted_dfdict,ensure_ascii=False,indent=1))

    def countPMIWindows(self, window_size=15, slide_length=1, padding=0,sample_size = 400):
        # 该方法计算的是单词之间的窗口共现情况
        # 记录格式为{'word': {'self': XXX(num) ,'cooccurence':{xxx:0,xxx:15,xxx:16....}}...}
        dataset = open(os.path.join(self.BaseDir,'GCN_dataset.json'), 'r', encoding='utf-8')
        # 建立字典记录各词所在窗口数 以及 该词与其他词共现窗口数
        cooccurence = dict()
        # 用于记录整个数据集的窗口总数
        windows_counts = 0
        for index, sample in tqdm(enumerate(dataset)):
            # 加载当前犯罪事实描述
            words = json.loads(sample.strip())['fact']
            # 开始记录自己出现的次数和共现次数
            start_index = 0
            while True:
                window_words = words[start_index:start_index+window_size]

                # 如果到头了，跳出
                # 需要保证是滑倒头的，而不是一开始文本长度就小于窗口长度
                if window_words[-1] == 0 and start_index != 0:
                    break
                # 数据集窗口总数+1
                windows_counts += 1

                # 起始索引后移
                start_index += slide_length

                # 循环记录, set去重
                cur_word_set = list(set(window_words))

                # 这个判断应该是没有用的（不过还是写在这里）
                if 0 in cur_word_set:
                    cur_word_set.remove(0)

                # 首先包含各词自身的window数加1
                for word in cur_word_set:
                    if word in cooccurence:
                        cooccurence[word]['self'] += 1
                    else:
                        cooccurence[word] = dict()
                        cooccurence[word]['self'] = 1
                        cooccurence[word]['cooccurence'] = dict()

                # 然后开始计算共现
                for w1_index,word in enumerate(cur_word_set):
                    # 内循环从当前词下一个开始，不然会重复计数
                    for w2_index in range(w1_index+1,len(cur_word_set)):
                        # 外层词与内层词共现窗口加1
                        # 对称的，所以都加
                        if cur_word_set[w2_index] not in cooccurence[word]['cooccurence']:
                            cooccurence[word]['cooccurence'][cur_word_set[w2_index]] = 1
                        else:
                            cooccurence[word]['cooccurence'][cur_word_set[w2_index]] += 1

                        if word not in cooccurence[cur_word_set[w2_index]]['cooccurence']:
                            cooccurence[cur_word_set[w2_index]]['cooccurence'][word] = 1
                        else:
                            cooccurence[cur_word_set[w2_index]]['cooccurence'][word] += 1

                # 如果读到了样本尽头也结束，
                # 这个情况针对于，犯罪事实大于400词，所以不会有0填充
                if start_index==sample_size-window_size+1:
                    break


        with open(os.path.join(self.BaseDir,f'GCN_PMIWindows_{windows_counts}.json'),
                  'w', encoding='utf-8') as f:
            f.write(json.dumps(cooccurence,ensure_ascii=False,indent=1))

        # 返回总窗口数
        print(f"窗口总数为{windows_counts}")
        return windows_counts

if __name__ == '__main__':
    tool = tools(BaseDir=f'D:\College Courses 2019.3-2019.6\信息管理课设\code\data\GCN\middleTest')
    # tool.constructOnehotFeatures()
    tool.constructLabels()

    # t = tool.constructLabels()
#     # coo = sparse.csc_matrix(sparse.load_npz(r'D:\College Courses 2019.3-2019.6\信息管理课设\code\data\GCN\smallerTest\onehotFeatures.npz'),shape=(8681,8681))
#     # print(coo)
#
    # labels = np.load(r'D:\College Courses 2019.3-2019.6\信息管理课设\code\data\GCN\smallerTest\labels.npy')
    # print(labels.shape)
    # print(labels)
#     pass
    pass