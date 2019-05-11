#encoding : utf-8
import scrapy
import scrapy_splash
import execjs
import uuid
import re
import json
from scrapy import Spider, Request
from scrapy_splash import SplashRequest


class WenshuSpider(scrapy.Spider):
    name = 'wenshu'
    allowed_domains = ['wenshu.court.gov.cn', 'wenku.jwzlai.com']
    start_urls = ['http://wenshu.court.gov.cn/']

    def parse(self, response):
        """
        从网站首页http://wenshu.court.gov.cn/提取所有案件分类
        :param response:
        :return:
        """
        all_types = response.xpath('//div[@id="nav"]/ul/li/a[@target="_blank"]')
        for ws_type in all_types[1:2]:
            # 获取a的文本内容：案件类型
            wenshu_type = ws_type.xpath('./text()').extract_first('').strip()
            # 再获取a的链接，拼接分类列表页的url。
            list_url = ws_type.xpath('./@href').extract_first('')
            # 拼接完整地址：http://wenshu.court.gov.cn/List/List?sorttype=1&conditions=searchWord+1+AJLX++案件类型:行政案件
            abs_url = 'http://wenshu.court.gov.cn' + list_url
            # 构造请求，向所有分类发送请求
            yield scrapy.Request(
                url=abs_url,
                callback=self.parse_list_page,
                meta={
                    'wenshu_type': wenshu_type
                }
            )

    def parse_list_page(self, response):
        """
        解析分类列表页：行政案件/民事案件
        :param response:
        :return:
        """
        # 在当前分类列表页请求成功以后，响应头中会有vjkl5这个cookie，要获取这个Cookie，后续需要传给js文件中的getKey()函数，通过vjkl5来获取vl5x这个参数。
        # getlist()获取'Set-Cookie'这个键对应的列表。
        vjkl5 = response.headers.getlist('Set-Cookie')[0].decode().split(';')[0].split('=')[1]
        # print(vjkl5)

        # 根据vjkl5获取vl5x的值，需要执行js函数getKey()
        # pip install PyExecJS

        js_content = open('C:/Users/Administrator/Desktop/wenshu.js', encoding='utf8').read()

        # js_obj = execjs.compile(js_content)返回一个js对象，该对象包含了js运行的环境。

        # 再使用js对象调用call()函数。
        # call('getKey', vjkl5): 参数1-要执行的js文件中对应的函数，参数2-给调用函数传递的参数。
        vl5x = execjs.compile(js_content).call('getKey', vjkl5)

        # js_content = open('C:/Users/Administrator/Desktop/wenshu.js', encoding='utf8').read()
        # guid = execjs.compile(js_content).call('get_guid') + execjs.compile(js_content).call('get_guid') + '-' + execjs.compile(js_content).call('get_guid')+ '-' + execjs.compile(js_content).call('get_guid') + execjs.compile(js_content).call('get_guid') + '-' + execjs.compile(js_content).call('get_guid') + execjs.compile(js_content).call('get_guid') + execjs.compile(js_content).call('get_guid')

        # 获取另外一个参数guid，可以通过uuid这个包来获取值。
        # uuid/md5/base64/sha1 是不同的加密包
        guid = str(uuid.uuid1())

        # 向内容接口http://wenshu.court.gov.cn/List/ListContent发送POST请求。
        url = 'http://wenshu.court.gov.cn/List/ListContent'
        post_data = {
            'Param': '案件类型:{}'.format(response.meta['wenshu_type']),
            'Index': '1',
            'Page': '5',
            'Order': '法院层级',
            'Direction': 'asc',
            'number': 'wens',
            'guid': guid,
            'vl5x': vl5x
        }

        # FormRequest: 发送POST请求的类。
        # Request: 默认发送GET请求，method设置请求方法。
        yield scrapy.FormRequest(
            url=url,
            callback=self.parse_list_json,
            formdata=post_data,
            # 由于在请求列表页接口时，cookie中vjkl5的值每次都是变化的，所以在请求时，需要将从response的Set-Cookie中获取的vjkl5的值更换一下，否则使用同一个vjkl5的值，会出现 'remind key' 错误。

            # urllib/requests: Cookie的自动化管理(cookiejar)？为什么要自动化管理，而不手动去解析？
            # 1. 复杂，每一个请求的响应都要去提取并存储；
            # 2. 容易遗漏；

            # scrapy框架是如何管理Cookie的？
            # 默认启用了一个Cookie的中间件，实现了Cookie的自动化管理。将所有响应的Set-Cookie中的cookie信息保存下来，在后续的请求中，将这些cookie携带上。

            # 为什么单独添加Cookies?
            # 因为每一次请求列表页，都会返回一个新的Set-Cookie: vjkl5=，但是scrapy请求时自动携带的cookie，可能还是之前的旧的vjkl5=，就会导致这个POST请求携带的Cookie无法和服务器保存的Cookie不一致。

            # 设置上cookies，就意味着每次请求都携带最新返回的vjkl5的值。
            cookies={
                'vjkl5': vjkl5
            }

            # 在使用cookie的时候，观察cookie的值的变化。
        )

    def parse_list_json(self, response):
        """
        解析列表页接口返回的JSON数据。
        :param response:
        :return:
        """
        # 利用正则表达式，将[]中的内容获取出来，是一个字典
        json_string = re.search(re.compile(r'\[(.*?)\]', re.S), response.text)
        if json_string:
            # 将字符串中 '\' 都替换为空。
            # \是特殊字符，需要表达普通字符\，需要使用\进行转义。
            result = json_string.group(1).replace("\\", '').replace('＆ｌｄｑｕｏ;', '（').replace('＆ｒｄｑｕｏ;', '）')

            # result是一个字符串 "{'RunEval':''},{'裁判要旨':'', '文书ID':''},{'裁判要旨':'', '文书ID':''},{'裁判要旨':'', '文书ID':''}"
            # 获取RunEval的值，加密字符串。
            runEval = re.search(re.compile(r'"RunEval":"(.*?)","', re.S), result).group(1)
            doc_ids = re.findall(re.compile(r'"文书ID":"(.*?)","', re.S), result)

            # 请求DOCID接口，对文书ID进行解密
            doc_url = 'http://wenku.jwzlai.com/common/decode/docId'

            post_data = {
                'runEval': runEval,
                'docIds': ','.join(doc_ids)
            }

            # 发送破解的POST请求
            yield scrapy.FormRequest(
                url=doc_url,
                callback=self.parse_pojie,
                formdata=post_data
            )


            # res = re.findall(re.compile(r'"文书ID":"(.*?)","', re.S), result)
            # # 上述result变量保存的就是列表页返回的json数据
            # # {"文书title", '文书id', '文书简介'}
            # for r in res:
            #     js_content = open('C:/Users/Administrator/Desktop/w.js', encoding='utf8').read()
            #     realID = execjs.compile(js_content).call('get_docid', r)
            #     # 请求详情页, DocID是唯一的，一个案件对应一个。
            #     detail_url = 'http://wenshu.court.gov.cn/CreateContentJS/CreateContentJS.aspx?DocID={}'.format(realID)
            #     yield scrapy.Request(
            #         url=detail_url,
            #         callback=self.parse_detail_page
            #     )

    def parse_pojie(self, response):
        """
        从识别结果中，遍历出所有的解密后的docid
        :param response:
        :return:
        """
        all_docids = json.loads(response.text)['data'].values()
        for docid in all_docids:
            # 根据docid拼接详情页地址
            detail_url = 'http://wenshu.court.gov.cn/CreateContentJS/CreateContentJS.aspx?DocID={}'.format(docid)
            yield scrapy.Request(
                url=detail_url,
                callback=self.parse_detail_page
            )

    def parse_detail_page(self, response):
        print(response.text)