import requests
import re
import execjs
import time
import json

getdocid_js = open('getDocid.js', 'r',encoding='utf-8').read()

class getListContend():

    def __init__(self,accusation,year=None):
        self.session = requests.session()
        if not accusation:
            self.param = '案件类型:刑事案件,文书类型:判决书,审判程序:一审'
        else:
            self.param = f'案件类型:刑事案件,文书类型:判决书,审判程序:一审,案由:{accusation}'
        if year is not None:
            self.param = f'案件类型:刑事案件,文书类型:判决书,审判程序:一审,案由:{accusation},裁判年份:{year}'
        #本次搜索结果总数,便于后面设置翻页操作
        self.count = 0

        #请求头
        self.headers = {
            'Upgrade-Insecure-Requests': "1",
            'Connection': 'keep-alive',
            'Host': 'wenshu.court.gov.cn',
            'Cache-Control': 'max-age=0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept - Encoding': 'gzip, deflate',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36",
        }

        #cookie中本次会话特别的参数,在getlistcontend中会被赋值
        self.cookie = {}

        #请求list的参数
        self.body = {'Param': self.param,
                'Index': '1',
                'Page': '10',
                'Order': '法院层级',
                'Direction': 'asc',
                'vl5x': '',#需要后面赋值,在getlistcontend中赋值后不会再改变
                'number': '',#需要后面赋值
                'guid': ''}#需要后面赋值

        #createguid js代码
        self.createGuid = '''
                            function getGuid() {
                                var guid = createGuid() + createGuid() + "-" + createGuid() + "-" + createGuid() + createGuid() + "-" + createGuid() + createGuid() + createGuid();
                                return guid
                            }
                                var createGuid = function () {
                                    return (((1 + Math.random()) * 65536) | 0).toString(16).substring(1)
                                }
                                    '''
        #生成翻页的number js代码
        self.createNumber = '''
                            function getNumber() {
                                var number = Math.random();
                                return number
                            }
                            '''

    def get_proxy(self):
        while True:
            while True:
                try:
                    print('请求代理中')
                    proxy_data = requests.get(
                    'http://www.xiongmaodaili.com/xiongmao-web/api/glip?secret=7e540b6dc1a99e673b3b3ae8d565a2d0&orderNo=GL201903242028442Z22vmzY&count=1&isTxt=0&proxyType=1',timeout = 10)
                    print('代理请求完成')
                    break
                except:
                    print('请求代理异常')
                    pass
            #while True:
            try:
                proxy_json = json.loads(proxy_data.text)
                ip = proxy_json['obj'][0]['ip']
                port = proxy_json['obj'][0]['port']
                ip_port = ip + ':' + port
                proxies = {
                    "http": "http://{}".format(ip_port),
                    "https": "http://{}".format(ip_port),
                }
                break
            except:
                print(f'代理数据异常 {proxy_data.text}')
                continue
        return proxies

    def getDocList(self):
        # 刑事案件
        spider_url = "http://wenshu.court.gov.cn/List/List?sorttype=1"
        #初始代理
        cur_proxies = self.get_proxy()
        print(f'初始代理{cur_proxies}')
        while True:
            try:
                print('首页开始请求')
                ret = self.session.get(spider_url, headers=self.headers,proxies = cur_proxies,timeout = 10)
                print('首页请求成功')
                try:
                    ret.encoding = 'utf-8'
                    _dict = ret.cookies.get_dict()
                    print('cookie 获得')

                    _sub = re.findall("eval\((.*?)\{\}\)\)", ret.text).pop()
                    _sub = 'function get_js(){return ' + _sub + "{})};"
                    print('js1完成')

                    after_js = execjs.compile(_sub).call("get_js")
                    _hock = "function init() { var ret = []; ret.push(dynamicurl); var wzwstemplate = KTKY2RBD9NHPBCIHV9ZMEQQDARSLVFDU(template.toString()); ret.push(wzwstemplate); var _wzwschallenge = QWERTASDFGXYSF() ; var wzwschallenge = KTKY2RBD9NHPBCIHV9ZMEQQDARSLVFDU(_wzwschallenge); ret.push(wzwschallenge); return ret;}"
                    _js2 = after_js.replace("HXXTTKKLLPPP5();", "") + _hock
                    print('js2完成')

                    ret = execjs.compile(_js2).call("init")

                    wzws_cid = _dict.get("wzws_cid")
                    print('wzws_cid 获得')
                    break
                except:
                    print('/List/List异常')
                    continue
                #break
            except:
                print('请求失败，代理异常，更换代理')
                cur_proxies = self.get_proxy()
        #while True:


        time.sleep(1)
        cookies = {
            "wzws_cid": wzws_cid,
            "wzwstemplate": ret[1],
            "wzwschallenge": ret[2],
        }

        url = "http://wenshu.court.gov.cn{}".format(ret[0])
        while True:
            try:
                print('开始请求新的wzws_cid')
                ret = self.session.get(
                    url,
                    headers=self.headers,
                    cookies=cookies,
                    allow_redirects=False,
                    proxies=cur_proxies,
                    timeout=10
                )
                wzws_cid = ret.cookies.get_dict()['wzws_cid']
                print('新的wzws_cid完成')
                self.cookies = {
                    'wzws_cid': wzws_cid
                }
                break
            except:
                print('请求失败，代理异常，更换代理')
                cur_proxies = self.get_proxy()
            #except:
            #    continue
        #print(self.cookies)
        # cookies.update(_gscu)
        while True:
            try:
                ret = self.session.get(spider_url, headers=self.headers, cookies=self.cookies,proxies = cur_proxies,timeout=10)
                # vjkl5
                vjkl5 = ret.cookies.get_dict()['vjkl5']
                #print(vjkl5)
                break
            except:
                print(f'{ret.status_code}...vjkl5 未成功')
                pass

        with open('getkey_final.js', 'r') as f:
            vl5x_js = f.read()
        ctx = execjs.compile(vl5x_js)
        while True:
            try:
                vl5x = ctx.call('getKey', vjkl5)
                break
            except:
                continue
        #print(vl5x)

        guid = execjs.compile(self.createGuid).call('getGuid')

        print(self.param)
        #更改body的三个参数
        self.body['vl5x'] = vl5x #当前参数会一直被使用
        self.body['number'] = 'wens'
        self.body['guid'] = guid

        docslist = self.buNaDaoShuJuJueBuReturn(1)

        return docslist

    def next_page(self):
        #设置需要执行的翻页次数
        next_times = 0
        if self.count==0:
            return []

        if self.count>=200:
            next_times = 19
        else:
            multiple = int(self.count/10)
            is_Remainder = self.count%10
            #十的整数倍的时候少翻一页
            if is_Remainder!=0:
                next_times = multiple
            else:
                next_times = multiple - 1

        for next_ in range(next_times):

            #v15x不再改变，翻页改变number和guid 以及改变index
            cur_number = execjs.compile(self.createNumber).call('getNumber')
            cur_guid = execjs.compile(self.createGuid).call('getGuid')
            cur_index = str(int(self.body['Index'])+1)
            #if cur_index == '4':
            #    continue
            #赋值
            self.body['Index'] = cur_index
            self.body['number'] = cur_number
            self.body['guid'] = cur_guid
            yield self.buNaDaoShuJuJueBuReturn(cur_index)

    def buNaDaoShuJuJueBuReturn(self,cur_index):
        content_link = "http://wenshu.court.gov.cn/List/ListContent"
        proxies = self.get_proxy()
        print(f'本次请求代理{proxies}')

        #将要返回的本次list数据
        docid_List = []
        count_0_times = 0
        #循环到可以return 不然不停
        while True:
        # 最多十次脏数据，不然换代理重请求
            for dirty_times in range(15):
                print('开始循环（单次循环限制十次脏数据）')
                #最内层循环，先正常访问网站否则换代理，且状态码200时弹出循环
                while True:
                    try:
                        print('请求中')
                        list_content = self.session.post(content_link, data=self.body, headers=self.headers,
                                                         cookies=self.cookies, proxies=proxies,timeout = 10)
                        print('请求结束')
                        # 确定不是502等
                        if list_content.status_code != 200:
                            print(f'请求状态码：{list_content.status_code},重新访问中')
                            if list_content.status_code == 503:
                                print('服务器过载,稍等一会儿')
                                time.sleep(10)
                                proxies = self.get_proxy()
                                print(f'本次请求代理{proxies}')
                        else:
                            print(f'{self.param} 第{cur_index}页 请求状态码：{list_content.status_code}')
                            break
                        #break
                    except:
                        print('代理异常，更换代理')
                        proxies = self.get_proxy()
                        print(f'本次请求代理{proxies}')

                #正则匹配Runeval 和 count 判断是否为脏数据
                json_list_str = json.loads(list_content.text)
                regex = '\"RunEval\":\"(?P<RunEval>.*?)\",\"Count\":\"(?P<count>.*?)\"'
                matcher = re.compile(regex, re.S)
                runeval_count = re.search(matcher, json_list_str)
                try:
                    RunEval = runeval_count.group('RunEval')
                    count = int(runeval_count.group('count'))
                except:
                    print('脏数据，无法正常得到runeval和count,重新请求')
                    continue
                #开始判断是否为脏数据
                #原数据转义过两次，这里先load一次，判断是否为正常list数据
                if json_list_str.endswith('},]'):
                    #如果是首页，多次判断是不是真地没结果
                    #如果不是首页，必定为脏数据
                    if cur_index == 1:
                        #如果尝试了超过5次，就认为确实没结果，不然再次请求
                        if count == 0 and count_0_times == 8:
                            #count_0_times = 0
                            print('当前检索条件，结果为0')
                            self.count = 0
                            return []
                        else:
                            count_0_times += 1
                            continue
                    else:
                        print('当前返回为脏数据,没有包含docid,重新请求')
                        #重新请求
                        continue
                # 加载了后，有的runeval值是错的，如果是错的就重新请求
                #elif not RunEval.startswith('w61'):
                #    print('当前返回为脏数据,重新请求')
                #    continue
                else:
                # 有时数据没办法加载，所以如果不行就重新请求
                    try:
                        jsondata_list = json.loads(json_list_str)
                        if not jsondata_list:
                            continue
                        # 设置self.count的值
                        self.count = int(jsondata_list[0]['Count'])
                        print(f' 本条件下总数：{self.count}')
                        if self.count ==0:
                            return []
                    except:
                        continue
                #如果能到达此步，则得到正常的list，接下来只需要解析不出错
                for i in range(1, len(jsondata_list)):
                    #必须要是能正常解析的runeval和docid，不然清空docid_list,然后重新请求
                    try:
                        docid = execjs.compile(getdocid_js).call('getdocid', RunEval, jsondata_list[i]['文书ID'])
                        if docid:
                            docid_List.append(docid)
                        else:
                            break
                    except:
                        docid_List.clear()
                        break
                #如果没有break,代表docid全部正确解析，返回list
                else:
                    print(f'{self.param} 第{cur_index}页 成功返回数据')
                    print(docid_List)
                    if docid_List:
                        return docid_List
                    else:
                        continue
            proxies = self.get_proxy()
            print(f'本次请求代理{proxies}')

if __name__ == '__main__':
    accusations = [
        '非法采伐、毁坏国家重点保护植物'
#'伪造、变造金融票证',
#'出售、购买、运输假币',
#'非法买卖制毒物品',
#'帮助毁灭、伪造证据',
#'窝藏、转移、隐瞒毒品、毒赃'
#'对单位行贿',
#'引诱、教唆、欺骗他人吸毒',
#'徇私枉法',
#'非法出售发票',
#'遗弃',
#'侵占',
#'走私珍贵动物、珍贵动物制品',
#'动植物检疫徇私舞弊',
#'聚众扰乱公共场所秩序、交通秩序',
#'拐骗儿童',
#'编造、故意传播虚假恐怖信息'
# '介绍贿赂',
# '非法收购、运输盗伐、滥伐的林木'
# '单位受贿',
# '生产、销售伪劣农药、兽药、化肥、种子',
# '走私废物',
# '帮助犯罪分子逃避处罚',
# '私分国有资产',
# '虚报注册资本',
# '非法进行节育手术',
# '伪证',
# '组织、领导、参加黑社会性质组织',
# '过失以危险方法危害公共安全',
# '过失投放危险物质',
# '过失损坏广播电视设施、公用电信设施',
# '招收公务员、学生徇私舞弊',
# '非法生产、买卖警用装备',
# '虐待',
# '收买被拐卖的妇女、儿童',
# '传授犯罪方法',
# '危险物品肇事',
# '盗窃、侮辱尸体',
# '非法制造、买卖、运输、储存危险物质',
# '传播性病',
# '走私武器、弹药',
# '盗窃、抢夺枪支、弹药、爆炸物',
# '破坏计算机信息系统',
# '破坏监管秩序',
# '强迫卖淫',
# '非法生产、销售间谍专用器材',
# '非法制造、销售非法制造的注册商标标识',
# '伪造、变造、买卖武装部队公文、证件、印章',
# '走私国家禁止进出口的货物、物品',
# '非法制造、出售非法制造的发票',
# '利用影响力受贿',
# '逃税',
# '串通投标'
#'聚众冲击国家机关',
#'诽谤',
#'伪造、倒卖伪造的有价票证',
#'挪用特定款物',
#'徇私舞弊不移交刑事案件',
#'非法获取国家秘密'
#'破坏交通设施',
#'巨额财产来源不明',
#'非法组织卖血',
#'侮辱',
#'伪造货币',
#'强迫劳动'
#'非法买卖、运输、携带、持有毒品原植物种子、幼苗', #没有这个罪名
#'金融凭证诈骗',
#'过失损坏武器装备、军事设施、军事通信',
#'非法收购、运输、出售珍贵、濒危野生动物、珍贵、濒危野生动物制品',
#'非法猎捕、杀害珍贵、濒危野生动物',
#'打击报复证人',
#'提供侵入、非法控制计算机信息系统程序、工具',
#'劫持船只、汽车',
#'洗钱',
#'徇私舞弊不征、少征税款',
#'倒卖车票、船票',
#'聚众哄抢',
#'破坏交通工具',
#'高利转贷',
#'倒卖文物',
#'虐待被监管人',
#'走私'
]
    for accusation in accusations:
        years = ['2019','2018','2017','2016','2015','2014','2013']
        #years = ['2014','2013']
        docid = open(f'yisheng/{accusation}-一审.txt', 'a+', encoding='utf-8')
        for year in years:
        #print(year)
            run = getListContend(accusation,year=year)
            #run.getDocList()
            #getdocid_js = open('getDocid.js', 'r',encoding='utf-8').read()
            for Docid_real in run.getDocList():
                print(Docid_real)
                #Docid_real= execjs.compile(getdocid_js).call('getdocid', runeval_doc[0],runeval_doc[1])
                docid.write(Docid_real+'\n')

            for Docid_real_list in run.next_page():
                #Docid_real = execjs.compile(getdocid_js).call('getdocid', page[0], page[1])
                for Docid_real in Docid_real_list:
                    print(Docid_real)
                    docid.write(Docid_real + '\n')
        docid.close()       #del run
