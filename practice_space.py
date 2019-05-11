import requests

headers = {

'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36',
#'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
'Accept-Encoding': 'gzip, deflate',
'Referer': 'http://wenshu.court.gov.cn/List/List?sorttype=1&conditions=searchWord+1+AJLX++%E6%A1%88%E4%BB%B6%E7%B1%BB%E5%9E%8B:%E8%A1%8C%E6%94%BF%E6%A1%88%E4%BB%B6',
'Connection': 'keep-alive',
'Upgrade-Insecure-Requests': '1',
'Cache-Control': 'max-age=0'
#'Cookie': '_gscu_2116842793=53358612qbm39v15; wzws_cid=1a86eea5fe5caa4ab8d0f58de8ffe6a5e599f2588450111e1f6b0eb8828f18dfe1e9afe8c2a06d758d513b279929a0e7; Hm_lvt_d2caefee2de09b8a6ea438d74fd98db2=1553358613,1553512131; _gscbrs_2116842793=1; vjkl5=197fbe696ff8ab4a901da189c27935a67ab7a87d; _gscs_2116842793=53512134r65p8t12|pv:2; Hm_lpvt_d2caefee2de09b8a6ea438d74fd98db2=1553512166'}
}
response = requests.get('http://wenshu.court.gov.cn/List/List?sorttype=1&conditions=searchWord+1+AJLX++案件类型:行政案件')
print(response.cookies)