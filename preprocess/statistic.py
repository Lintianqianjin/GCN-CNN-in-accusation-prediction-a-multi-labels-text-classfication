import json
def crime_articles_distribution():
    data4 = open('../data/data.json','r',encoding='utf-8')
    distribution_accusation = open('../result/罪名分布.txt','w',encoding='utf-8')
    distribution_articles = open('../result/法条分布.txt','w',encoding='utf-8')

    crime_dict = {}
    article_dict = {}

    for index,line in enumerate(data4):
        print(index)
        cur_data = json.loads(line.strip())
        articles= cur_data['meta']['relevant_articles']
        accusation = cur_data['meta']['accusation']

        for a in articles:
            if a not in article_dict:
                article_dict[a] = 1
            else:
                article_dict[a] += 1
        for ac in accusation:
            ac = ac.replace('[', '').replace(']', '')
            if ac not in crime_dict:
                crime_dict[ac] = 1
            else:
                crime_dict[ac] += 1

    for key in crime_dict:
        distribution_accusation.write(key+'\t'+str(crime_dict[key])+'\n')

    distribution_accusation.close()

    for article in article_dict:
        distribution_articles.write(str(article)+'\t'+str(article_dict[article])+'\n')
    distribution_articles.close()

def sentence_length():
    pure_id = open(r'../data/词ID-罪名ID-法条.json', 'r', encoding='utf-8')
    len_dict = {}
    for index,line in enumerate(pure_id):
        print(index)
        cur = json.loads(line.strip())
        cur_len = str(len(cur['fact']))
        if cur_len not in len_dict:
            len_dict[cur_len] = 1
        else:
            len_dict[cur_len] += 1
    with open(r'../result/犯罪事实描述长度.txt','w') as f:
        for sentence_len in len_dict:
            f.write(sentence_len+'\t'+str(len_dict[sentence_len])+'\n')


if __name__ == '__main__':
    sentence_length()













