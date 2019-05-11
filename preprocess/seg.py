import jieba
import regex as re
import json
import time
import os
import multiprocessing as mp

index = 0

def merge_dict():
    laws = open(r'../data/搜狗法律词典/法律词汇大全【官方推荐】.txt',encoding='utf-8')
    locs = open(r'../data/搜狗法律词典/全国省市区县地名.txt',encoding='utf-8')
    new_diy_dict = open(r'../data/搜狗法律词典/自定义词典.txt','w',encoding='utf-8')

    diy_dict = {}
    index = 1

    for line in laws:
        if line.strip() not in diy_dict:
            diy_dict[line.strip()] = ''
        print(index)
        index+=1

    for line in locs:
        if line.strip() not in diy_dict:
            diy_dict[line.strip()] = ''
        print(index)
        index += 1

    for word in diy_dict:
        new_diy_dict.write(word+'\n')

#加载停用词表进内存
def load_stopwords():
    stops = open(r'D:\College Courses 2019.2-2019.6\信息管理课设\code\data\搜狗法律词典\停用词', encoding='utf-8')
    stop_dict = {}
    for line in stops:
        if line.strip() not in stop_dict:
            stop_dict[line.strip()] = ''

    special_list = ['\n','\t','\r','\r\n']
    for spec in special_list:
        stop_dict[spec] = ''

    return stop_dict

#去除停用词，返回取出后的句子分词列表
def delete_stopwords(seglist,stops):
    print(time.time())
    #seglist 是分好词的句子的列表
    #stops 是停用词表，类型为字典
    return list(filter(lambda x: x in stops, seglist))

#对一个句子进行分词
def segment_sentence(sentence):
    jieba.load_userdict(r'../data/搜狗法律词典/自定义词典.txt')
    seged = jieba.cut(sentence,cut_all=False)
    return list(seged)

def seg_file_into_new(chunkStart, chunkSize):
    #data_source =
    #new_file = open(r'../data/分词事实-罪名-法条.json','w',encoding='utf-8')
    stpw = load_stopwords()
    jieba.load_userdict(r'../data/搜狗法律词典/自定义词典.txt')

    with open(r'../data/data.json','rb') as f:
        #for index,line in enumerate(data_source):
        #print(index)
        #print(time.time())
        f.seek(chunkStart)
        lines = f.read(chunkSize).decode('utf-8').splitlines()
        dataset = []
        for line in lines:
            #print(line)
            cur_data = json.loads(line.strip())
            articles = cur_data['meta']['relevant_articles']
            accusation = cur_data['meta']['accusation']
            sentence = cur_data['fact']
            #print(time.time())
            seglist = jieba.cut(sentence, cut_all=False)
            #print(time.time())
            fact = list(filter(lambda x: x not in stpw, seglist))
            #print(time.time())
            new = {'fact':fact,'accusation':accusation,'articles':articles}
            #print(time.time())
            #new_file.write()
            #print(time.time())
            #new_file.close()
            #break
            dataset.append(json.dumps(new,ensure_ascii= False)+'\n')
        return dataset

def chunkify(fname,size=1024*1024*8):
    #inedx = 1
    fileEnd = os.path.getsize(fname)
    with open(fname,'rb') as f:
        chunkEnd = f.tell()
        while True:
            chunkStart = chunkEnd
            #print(chunkStart)
            f.seek(size,1)
            #print(f.tell())
            f.readline()
            #print(f.tell())
            chunkEnd = f.tell()
            yield chunkStart, chunkEnd - chunkStart
            if chunkEnd > fileEnd:
                break

def write_data_callback(datas):
    global index
    with open('../data/分词事实-罪名-法条.json', 'a+',encoding='utf-8') as f:
        for data in datas:
            #print('writing')
            print(index)
            index+=1
            f.write(data)

def multi_seg_main():
    pool = mp.Pool(4)
    # m = mp.Manager()
    # q = m.Queue()
    # jobs = []
    for chunkStart, chunkSize in chunkify(r'../data/data.json'):
        #print('aaa')
        try:
            pool.apply_async(seg_file_into_new, args=(chunkStart, chunkSize,), callback=write_data_callback)
        except Exception as e:
            print(e)
            # res = []
            # for job in jobs:
            # res.append(job.get())
    pool.close()
    pool.join()

def word_count_file_appearence():
    words = {}
    data = open(r'../data/分词事实-罪名-法条.json','r',encoding='utf-8')
    result = open(r'../result/词出现的文档数.txt','w',encoding='utf-8')
    for index,line in enumerate(data):
        print(index)
        cur_crime = json.loads(line.strip())
        #计算IDF，需要去重
        for word in set(cur_crime['fact']):
            if word not in words:
                words[word] = 1
            else:
                words[word] += 1
    write_index = 1
    for word in sorted(words.items(),key=lambda item:item[1],reverse=True):
        print(write_index)
        write_index+=1
        result.write(word[0]+'\t'+str(word[1])+'\n')

def word_idf_ranked():
    #选取出现次数小于200W，即排除被告人、年、月、日、指控
    words_times = open(r'../result/词出现的文档数.txt','r',encoding='utf-8')
    idf_high = open(r'../result/前8万词.txt','w',encoding='utf-8')
    words_dict = {}

    matcher = re.compile(r'\d+')

    for word_line in words_times:
        word = word_line.split('\t')[0]
        match = matcher.search(word)
        if not match:
            words_dict[word] = int(word_line.split('\t')[1].strip())

    rank_id = 1
    for word in sorted(words_dict.items(), key=lambda item: item[1], reverse=True):
        if word[1]>2000000:
            continue
        else:
            idf_high.write(word[0]+'\n')
            print(rank_id)
            rank_id+=1
            if rank_id > 80000:
                break

def word2id():
    data = open(r'../data/分词事实-罪名-法条.json', 'r', encoding='utf-8')
    word_id_data = open(r'../data/词ID-罪名-法条.json', 'w', encoding='utf-8')
    top_words = open(r'../result/前8万词.txt', 'r', encoding='utf-8')
    word2id_dict = {}

    #从1开始 后面0用来填充
    for index,word in enumerate(top_words):
        word2id_dict[word.strip()] = index+1

    for index,crime in enumerate(data):
        print(index)
        cur_crime = json.loads(crime.strip())
        ID_seq = []
        for w in cur_crime['fact']:
            try:
                id = word2id_dict[w]
                ID_seq.append(id)
            except:
                continue
        curjson = {'fact': ID_seq,'accusation':cur_crime['accusation'],'articles':cur_crime['articles']}
        word_id_data.write(json.dumps(curjson,ensure_ascii=False)+'\n')

def accusation2id():
    word_id_data = open(r'../data/词ID-罪名-法条.json', 'r', encoding='utf-8')
    accusations = open(r'../result/罪名分布.txt', 'r', encoding='utf-8')
    pure_id = open(r'../data/词ID-罪名ID-法条.json', 'w', encoding='utf-8')
    accusation2id_dict = {}

    for index, accusation in enumerate(accusations):
        accusation2id_dict[accusation.split('\t')[0]] = index

    for index, crime in enumerate(word_id_data):
        print(index)
        cur_crime = json.loads(crime.strip())
        accusation_id = [accusation2id_dict[x.replace(']','').replace('[','')] for x in cur_crime['accusation']]
        #cur_crime['accusation'] = accusation_id
        cur_data = {'fact':cur_crime['fact'],'accusation':accusation_id,'articles':cur_crime['articles']}
        pure_id.write(json.dumps(cur_data, ensure_ascii=False) + '\n')

def controled_length():
    pure_id = open(r'../data/词ID-罪名ID-法条.json', 'r', encoding='utf-8')
    improved_pure_id = open(r'../data/400词ID-罪名ID-法条ID.json', 'w', encoding='utf-8')
    for index, crime in enumerate(pure_id):
        print(index)
        cur_crime = json.loads(crime.strip())
        cur_fact = [word for word in cur_crime['fact']]
        fact_len = len(cur_fact)
        if fact_len < 400:
            blank = [80000 for i in range(400-fact_len)]
            new_fact = cur_fact + blank
        elif fact_len>400:
            new_fact = [cur_fact[index] for index in range(400)]
        else:
            new_fact = cur_fact
        cur_data = {'fact': new_fact, 'accusation': cur_crime['accusation'], 'articles': [int(art) for art in cur_crime['articales']]}
        improved_pure_id.write(json.dumps(cur_data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    pass