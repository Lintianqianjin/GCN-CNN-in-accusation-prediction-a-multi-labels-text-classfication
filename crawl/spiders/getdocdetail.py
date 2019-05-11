import json
import re
import execjs
import requests

import time

class Detail():
    def __init__(self):
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
    def getproxy(self):
        proxy_data = requests.get('http://www.xiongmaodaili.com/xiongmao-web/api/glip?secret=7e540b6dc1a99e673b3b3ae8d565a2d0&orderNo=GL201903242028442Z22vmzY&count=1&isTxt=0&proxyType=1',timeout = 10)
        proxy_json = json.loads(proxy_data.text)
        ip = proxy_json['obj'][0]['ip']
        port = proxy_json['obj'][0]['port']
        ip_port = ip + ':' + port
        return {"http": f"http://{ip_port}","https": f"http://{ip_port}"}

    def loopGET(self,spider_url=None,headers=None,cookies=None,allow_redirects=False,proxies=None,timeout = 10,session = None):
        if session is None:
            print('session为空！')
        else:
            while True:
                try:
                    if cookies is None:
                        response = session.get(spider_url, headers=headers, proxies=proxies,timeout = timeout)
                    else:
                        response = session.get(spider_url, headers=headers, proxies=proxies, timeout=timeout,cookies=cookies)
                    #status_code = response.status_code
                    #if status_code == 200:
                    return response
                except:
                    pass


    def getdetail(self,doc_id,proxies):
        session = requests.session()

        spider_url = f"http://wenshu.court.gov.cn/content/content?DocID={doc_id}&KeyWord="
        ret = self.loopGET(spider_url, headers=self.headers, proxies=proxies,timeout = 5,session=session)
        ret.encoding = 'utf-8'
        _dict = ret.cookies.get_dict()

        _sub = re.findall("eval\((.*?)\{\}\)\)", ret.text).pop()
        _sub = 'function get_js(){return ' + _sub + "{})};"

        after_js = execjs.compile(_sub).call("get_js")
        _hock = "function init() { var ret = []; ret.push(dynamicurl); var wzwstemplate = KTKY2RBD9NHPBCIHV9ZMEQQDARSLVFDU(template.toString()); ret.push(wzwstemplate); var _wzwschallenge = QWERTASDFGXYSF() ; var wzwschallenge = KTKY2RBD9NHPBCIHV9ZMEQQDARSLVFDU(_wzwschallenge); ret.push(wzwschallenge); return ret;}"
        _js2 = after_js.replace("HXXTTKKLLPPP5();", "") + _hock

        ret = execjs.compile(_js2).call("init")
        print(_dict.get("wzws_cid"))
        wzws_cid = _dict.get("wzws_cid")

        time.sleep(1)
        cookies = {
            "wzws_cid": wzws_cid,
            "wzwstemplate": ret[1],
            "wzwschallenge": ret[2],
        }

        url = f"http://wenshu.court.gov.cn{ret[0]}"
        ret = self.loopGET(url,headers=self.headers,cookies=cookies,allow_redirects=False,proxies=proxies,timeout=5,session=session)
        print(ret.cookies.get_dict())

        wzws_cid = ret.cookies.get_dict()['wzws_cid']

        spider_url = f"http://wenshu.court.gov.cn/content/content?DocID={doc_id}&KeyWord="
        cookies = { 'wzws_cid': wzws_cid }
        ret = self.loopGET(spider_url, headers=self.headers, cookies=cookies,timeout = 5)
        cookies = { "wzws_cid": wzws_cid }

        ret = self.loopGET(f"http://wenshu.court.gov.cn/CreateContentJS/CreateContentJS.aspx?DocID={doc_id}",
            headers=self.headers,
            cookies=cookies,
            proxies=proxies,
            timeout = 5,
            session = session
        )

        print(ret.text)

if __name__ == '__main__':
    run = Detail()
    run.getdetail(doc_id='1cd0afcc-8b89-4aa2-941e-aa0000bac78e',proxies=run.getproxy())