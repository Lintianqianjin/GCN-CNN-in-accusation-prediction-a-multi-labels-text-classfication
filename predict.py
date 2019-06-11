# encoding:utf-8
import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import GRU, MaxPooling1D, Bidirectional
import pandas as pd
import time
from keras.models import load_model
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

from model.manyModels import top1Accuracy,strictAccuracy
import preprocess.seg as seg
import jieba
from keras.models import load_model
from model.CNN import predict2top


def load_fact(fact):
    #分词并去停用词
    stpw = seg.load_stopwords()
    jieba.load_userdict(r'data\搜狗法律词典\自定义词典.txt')
    seglist = jieba.cut(fact, cut_all=False)
    fact_word_list = list(filter(lambda x: x not in stpw, seglist))

    #只保留前8W词
    top_words = open(r'result\前8万词.txt', 'r', encoding='utf-8')
    word2id_dict = {}
    # 从1开始 后面0用来填充
    for index, word in enumerate(top_words):
        word2id_dict[word.strip()] = index + 1
    ID_seq = []
    for w in fact_word_list:
        try:
            id = word2id_dict[w]
            ID_seq.append(id)
        except:
            continue

    #填充/截长到400词
    fact_len = len(ID_seq)
    if fact_len < 400:
        blank = [0 for i in range(400 - fact_len)]
        new_fact = ID_seq + blank
    elif fact_len > 400:
        new_fact = [ID_seq[index] for index in range(400)]
    else:
        new_fact = ID_seq

    return np.array([new_fact])

def predict(X):

    modelFiles = os.listdir('model/trainedModels')
    # accs = {'strictAccuracy': 0, 'top1Accuracy': 0}
    models = []
    for modelName in modelFiles:
        model = load_model(os.path.join('model\\trainedModels', modelName))
        models.append(model)

    print(f'{modelName} prediction start **********')

    Y_preds = [model.predict(X) for model in models]

    new_Y_pred = []

    for i in range(len(X)):
        cur_pred = 0
        for each_preds in Y_preds:
            cur_pred += each_preds[i]

        cur_pred = np.where(cur_pred >= 2, int(1), int(0))
        # print(cur_pred)
        new_Y_pred.append(cur_pred)
    print(f'{modelName} prediction end **********')

    id2name = {}
    file = open('result/罪名分布.txt','r',encoding='utf-8')
    for index,line in enumerate(file):
        id2name[index] = line.split('\t')[0]

    crime_list = []
    # 转ID为罪名
    for predict in new_Y_pred:
        crimeindexes = np.where(predict==1)[0]
        cur_crime_name = [id2name[crimeindex] for crimeindex in crimeindexes]
        crime_list.append(cur_crime_name)

    return crime_list

    # print(accs)

def accusation_id2name(one_hot_id):
    accusation_dict = {0:'盗窃',1:'故意伤害',2:'诈骗',3:'开设赌场',4:'危险驾驶',5:'抢劫',6:'容留他人吸毒',7:'信用卡诈骗'}
    acc_list = []
    for index,binary in enumerate(one_hot_id):
        if binary == 1:
            acc_list.append(accusation_dict[index])
    return acc_list

if __name__ == '__main__':
    # facts = ['经审理查明，2015年1月27日11时许，被告人马某某在邵东县城兴湘菜市场，趁申某某不注意之机，'
    #          '将申某某放在衣袋内的一台价值6300元的苹果6手机扒走。申某某报案后，公安民警根据被盗手机安装的定位软件，'
    #          '在邵东县城广场南路一手机维修店附近将准备销赃的马某某抓获，当场在马某某身上搜缴被盗手机，发还申某某。',
    #
    #         '浏阳市人民检察院指控，被告人姜某某曾系本市关口街道办事处金桥村“某某山庄”厨师，'
    #         '后于2015年9月15日从“某某山庄”辞职。被告人姜某某辞职后，于9月15日晚10时许独自驾车窜至“某某山庄”，'
    #         '溜门进入该山庄大厅后，将收银台上一台黑色“酷罗士”电脑一体机盗走。经价格鉴定，被盗电脑价值人民币2660元。',
    #
    #         '公诉机关指控，2014年11月28日15时40分，公安机关接被害人刘某报案称，2014年11月下旬，'
    #         '被害人刘某的餐厅急需招聘员工，被告人罗某称能介绍六名工人过来，但要给工人补工资及车费，'
    #         '2014年12月23日，被害人刘某在龙岗区某公寓一区630房给了被告人罗某13600元钱，作为六名工人的工资及车费。'
    #         '次日，被告人罗某携款潜逃。2015年5月2日，公安机关将被告人罗某抓获归案。',
    #
    #         '贺州市八步区人民检察院指控，2013年12月16日，被告人莫伟、李丽霞伙同蒙向紫、林远英（均另案处理）'
    #         '窜到贺州市八步城区城西市场附近的鼎富小区，由莫伟扮演医生负责物色作案目标，李丽霞扮演“何老师”，'
    #         '蒙向紫扮演老中医的孙子，林远英负责望风，先骗取被害人黄某某的信任，'
    #         '然后以黄某某的儿子近期将有血光之灾需要化解为由，骗取黄某某现金69800元。',
    #
    #         '九江市浔阳区人民检察院指控：2014年1月31日晚上19时30分许，曾某某驾驶车牌为鄂A31E**的丰田汉兰达越野车'
    #         '行驶至九江石化社区五区一栋旁，与周某某驾驶的车牌为赣G0T6**的现代伊兰特轿车发生剐蹭。'
    #         '双方下车后对车辆受损程度及赔偿问题意见不一致，即各自通知亲属到达现场。随后双方再次因为车辆赔偿问题发生争执，'
    #         '进而引发肢体冲突。在争执过程中，周某某的丈夫即被告人田某某将曾某某的亲属即被害人洪某某打伤。'
    #         '经鉴定，被害人洪某某的伤情为轻伤一级。',
    #
    #         '中山市第一市区人民检察院指控：2015年4月19日晚上11时许，被告人欧某某未取得机动车驾驶资格醉酒后'
    #         '（经鉴定，欧某某驾驶车辆时血液中乙醇含量为183.3mg／100ml）驾驶无号牌二轮摩托车由佛山市顺德区往中山市方向行驶，'
    #         '驶至中山市大涌镇古神公路时，换由王某某驾驶该摩托车继续驶至古神公路CA0737—0738灯柱路段时，'
    #         '与阳某某驾驶的湘D89V**号二轮摩托车发生碰撞，造成王某某死亡、多人受伤及车辆损坏。'
    #         '随后欧某某在医院被公安人员抓获，同年5月12日，欧某某到公安机关归案，并如实供述了上述罪行。'
    #         ]

    facts = [
        "公诉机关指控,一、2014年11月28日12时46分许,山东省临沂市兰山区人民法院执行局工作人员张某、"
        "刘某等人驾驶鲁Q?????警车到位于临沭县城北外环魏某乙的金山会所,"
        "对拒不履行生效的(2014)临兰民初字第2193号民事判决书的被告人魏某甲实施司法拘留。"
        "为逃避执行,魏某甲从会所院内驾驶鲁Q?????黑色上海大众途锐轿车,采取撞击的方式将停在院门口的警车强行逼挤至院外,"
        "致警车的保险杠损坏,后魏某甲驾车逃窜。执法人员令其父亲魏某乙上车,到金山化肥厂寻找魏某甲。"
        "车辆行驶至该公司门口时遇到魏某甲,执法人员遂驾车追赶,并令魏某乙电话联系魏某甲。魏某甲在得知其父在执法车辆上,"
        "遂电话联系了被告人周某,被告人王2某、李某也闻讯赶来。在327国道上,魏某甲驾驶鲁Q?????大众途锐轿车,"
        "周某驾驶鲁Q?????帕某轿车,王2某驾驶鲁Q?????丰某越野车,李某驾驶黑色别克轿车追逐、围困正在行驶中的警车,"
        "并多次采取冲撞、轧车的方式拦截执法警车,后将警车逼停。执法人员令魏某乙下车,魏某乙上魏某甲驾驶的车辆后,"
        "魏某甲将车头调转,欲对撞警车,因被魏某乙拨动方向盘未果。后魏某甲、周某、王2某、李某驾车离开现场。经鉴定,"
        "东风风行菱智(鲁Q?????)损失共计410元。二、2014年10月1日至2014年11月10日,被告人魏某甲在未取得采砂许可手续的情况下,"
        "雇佣他人在临沭县沭河河道朱村段非法开采黄某。经测量,被告人魏某甲雇佣他人非法开采的黄某达42000余方,价值1260000元。"
        "公诉机关认为,被告人魏某甲以暴力、威胁方式阻碍国家工作人员依法执行职务,其行为触犯了《中华人民共和国刑法》××的规定,"
        "应当以××追究其刑事责任。被告人魏某甲违反矿产资源法的规定,未取得采矿许可证擅自采矿,情节特别严重,其行为触犯了"
        "《中华人民共和国刑法》××××的规定,应当以××追究其刑事责任。",

        "四川省宜某市翠屏区人民检察院指控:被告人江1某已办理烟草专卖零售许可证,"
        " 被告人朱1某、何1某未办理烟草专卖零售许可证。2017年以来, 被告人朱1某从他人处购进"
        "假冒的云烟、玉溪、牡丹、中华、白塔山、南京、利群、中华、芙某、阿某等卷烟"
        "和少量正品红塔山恭贺新禧卷烟及少量走私的爱喜卷烟, 并伙同其妻子被告人何1某一同将购买的烟加价转卖给被告人江1某,"
        "由被告人朱1某通过成都到宜某的野的司机曹某红或快递将烟送到宜某市翠屏区交给被告人江1某,"
        " 被告人何1某负责记录销售情况,被告人朱1某提供了一张户名为黄某的建设银行卡给被告人何1某, "
        "用于专门向被告人江1某收取卷烟销售款。2017年7月至9月期间, "
        "被告人江1某通过银行存款的方式向被告人何1某持有的银行卡内转账共197850元用于支付购烟款,"
        "另在宜某市翠屏区人民公园后门支付被告人朱1某购烟款30000元, 上述款项中有2500元是用于支付正品红塔山恭贺新禧卷烟款,"
        "有4125元是用于支付走私爱喜卷烟款。被告人江1某除向被告人朱1某、何1某购买假烟外, 还向他人购买假烟, "
        "并一同放在其经营的位于宜某市翠屏区建设路134号的门市上进行零售或向杨某1、杨某2等人批发销售。"
        "被告人江1某已将从被告人朱1某、何3某购买的174010元的假烟予以销售, 销售金额20万余元。2017年9月30日, "
        "公安机关在建设路134号门市搜出被告人江1某尚未销售的假冒硬大前门35条、软大前门50条、"
        "软经典双喜4条、软牡丹67条、硬阿某15条、硬经典红塔山13条、软珍品云烟95条、软如意云烟40条、"
        "紫云烟35条、硬芙某84条、炫赫门南京10条、新版利群14条、软中华8条、硬中华17条、细支中华11条、"
        "硬玉溪82条、软玉溪24条、细支玉8条, 以上假烟购买价格共计46945元",
        "梁河县人民检察院指控:1、2014年7月21日10时20分,梁河县公安局民警到梁河县芒东镇翁冷村委会丙那二组"
        "被告人王2某家中将王2某抓获,当场从王2某家中查获海洛因5.1克、甲基苯丙胺17.5克。"
        "经查,被告人王2某除自己吸食毒品外,还向方某3甲、谷某某、金某某、郭某某贩卖过毒品海洛因。"
        "2、2015年6月18日10时左右,梁河县公安局民警到梁河县芒东镇翁冷村委会丙那后山被告人王2某的窝铺中将王2某抓获,"
        "当场从王3某铺中查获海洛因0.2克,甲基苯丙胺7克。经查,被告人王2某除自己吸食毒品外,还向马某、"
        "曹某、张某、方某3乙贩卖过毒品海洛因,向钱某某贩卖过毒品海洛因和甲基苯丙胺。"
        "3、2015年1月2日晚上曹某、马某、张某在梁河县翁冷村委会丙那后山王2某的窝铺里与王2某商量去偷羊和王2某换毒品吸食的事情,"
        "王2某同意后,2015年1月3日2时左右,曹某、马某、张某三人就驾驶摩托车到梁河县另电站旁被害人杨某1的羊圈。"
        "张某进入被害人杨某1的窝铺内用刀挟持杨某1,在杨某1不敢反抗的情况下,马某和曹某用摩托车拉走了杨某1的5头山羊。"
        "经鉴定,被盗山羊的价格为人民币5600元。被告人王2某对公诉机关指控的事实及罪名无异议,请求法庭从轻判处。"]


    facts_list = []

    for fact in facts:
        facts_list.append(list(load_fact(fact)))


    #print(fact)
    #print(fact)
    result = predict(np.array(facts_list).reshape((len(facts),400)))
    print(result)