#encoding : utf-8
import scrapy
import scrapy_splash
import execjs
import uuid
import re
import json
from scrapy import Spider, Request
from scrapy_splash import SplashRequest


class wenshuSpider(scrapy.spiders.Spider):
    name = 'phone'
    allowed_domains = ['www.baidu.com']
    url = 'https://www.baidu.com'

    # start request
    def start_requests(self):
        yield SplashRequest(self.url, callback=self.parse, endpoint='execute',
                            args={'lua_source': script, 'phone': '18707188761', 'wait': 5})

    # parse the html content
    def parse(self, response):
        info = response.css('div.op_mobilephone_r.c-gap-bottom-small').xpath('span/text()').extract()
        print('=' * 40)
        print(''.join(info))
        print('=' * 40)

