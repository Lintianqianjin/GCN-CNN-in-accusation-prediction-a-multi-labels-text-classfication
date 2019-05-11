import json
import regex as re
import os
import time

class parse_wensudoc():

    def __init__(self,dirpath):
        self.parsepath = dirpath
        self.regex_for_jsonHtmlData = '(jsonHtmlData = (?P<htmldata>.*?)var)'
        self.regex_for_htmldetail = '<div.*?>(?P<text>.*?)</div>'

    #解析爬取的js代码,返回文书标题和文书详情内容
    def parse_gethtmldetail(self):
        regex = re.compile(self.regex_for_jsonHtmlData,re.S)
        document = open(self.parsepath,'r',encoding='utf-8').read()
        #try:
        #time.sleep(1)
        detail = json.loads(json.loads(regex.search(document).group('htmldata').strip().strip(';'),strict = False),strict = False)
        #except:
        #print(document)
            #return None
        return (detail['Title'],detail['Html'].replace('\n','').replace('\t',''))

    #从文书详情内容的html文档中分离出犯罪事实、判决罪名、判刑时长、罚款
    def parse_get_format_data(self,htmldata):
        regex = re.compile(self.regex_for_htmldetail, re.S)
        texts = regex.findall(htmldata)
        return texts


if __name__ == '__main__':

    dir =r'D:\College Courses 2019.3-2019.6\信息管理课设\文书案例数小于500（中国裁判文书网）\非法制造、出售非法制造的发票'
    docs = os.listdir(dir)

    for doc in docs :
        docpath = os.path.join(dir,doc)
        test = parse_wensudoc(docpath)
        html = test.parse_gethtmldetail()[1]
        texts = test.parse_get_format_data(html)
        for paragraphy in texts:
            if '上述犯罪事实'in paragraphy or '上述事实' in paragraphy or '以上事实' in paragraphy or '以上犯罪事实' in paragraphy:
                break
        else:
            #time.sleep(1)
            print(doc)
        print(len(texts))