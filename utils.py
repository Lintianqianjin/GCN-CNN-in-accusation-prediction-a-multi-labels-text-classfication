import numpy as np
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def generate_X_Y_Data():
    data = open(r'../data/FD_AUG.json','r',encoding='utf-8')

    X = []
    Y = []

    for index, line in tqdm(enumerate(data)):
        # print(index)
        cur = json.loads(line.strip())
        X.append([int(word) for word in cur['fact']])
        one_hot = np.zeros(shape=(202))
        for i_acc in cur['accusation']:
            one_hot[i_acc] = 1
        Y.append(one_hot)

    X = np.array(X,dtype=int)
    Y = np.array(Y,dtype=int)

    print(len(X))
    print(len(Y))
    # exit()
    # np.savetxt(r'../data/final_X.txt', X,delimiter='\t',fmt='%d')
    # np.savetxt(r'../data/final_Y.txt', Y,delimiter='\t',fmt='%d')
    np.save('../data/FinalData/final_X',X)
    np.save('../data/FinalData/final_Y',Y)


def BatchDataGenerator(Batch_size):
    pass

def split_train_test_valid():
    X = np.load(r'D:\CollegeCourses2019.3-2019.6\信息管理课设\code\data\FinalData\final_X.npy')
    Y = np.load(r'D:\CollegeCourses2019.3-2019.6\信息管理课设\code\data\FinalData\final_Y.npy')
    # print(len(X))
    # print(len(Y))

    # exit()
    X_train, X_rest = train_test_split(X, test_size=0.07, random_state=123)
    Y_train, Y_rest = train_test_split(Y, test_size=0.07, random_state=123)

    X_test, X_valid = train_test_split(X_rest, test_size=0.5, random_state=520)
    Y_test, Y_valid = train_test_split(Y_rest, test_size=0.5, random_state=520)

    np.save('../data/FinalData/final_X_train', X_train)
    np.save('../data/FinalData/final_Y_train', Y_train)
    np.save('../data/FinalData/final_X_test', X_test)
    np.save('../data/FinalData/final_Y_test', Y_test)
    np.save('../data/FinalData/final_X_valid', X_valid)
    np.save('../data/FinalData/final_Y_valid', Y_valid)

def predict2top(predictions):
    one_hots = []
    for prediction in predictions:
        max = prediction.max()
        one_hot = np.where(prediction == max , int(1), int(0))
        one_hots.append(one_hot)
    return np.array(one_hots)

def plotCompareAccCurve():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    curveDataFile = os.listdir('accCurves')
    for fileName in curveDataFile:
        filepath = os.path.join('accCurves\\'+fileName)
        data = json.load(open(filepath,'r',encoding='utf-8'))
        plt.plot([i for i in range(1, len(data) + 1)], data, label=fileName.split('.')[0])
    plt.legend(fontsize = 14,borderpad=1)
    plt.xlabel('batches/50',fontsize = 18)
    plt.ylabel('测试集严格准确率',fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize=16)
    # todo: 更换模型时要改名字
    plt.savefig('AccCompare.png', dpi=300)

if __name__ == '__main__':
    plotCompareAccCurve()
    # X = np.load('../data/FinalData/final_Y_train.npy')
    # print('finish load')
    # print(len(X))
    # num = 0
    # for row in tqdm(X):
    #     if np.sum(row) > 1:
    #         num += 1
    # print(num)

    # split_train_test_valid()
    # Y = np.load('../data/FinalData/final_Y_train.npy')
    # print('finish load')
    # print(len(Y))
    # num = 0
    # for row in tqdm(Y):
    #     if np.sum(row) > 1:
    #         num += 1
    #
    # print(num)
    # generate_X_Y_Data()
    # split_train_test_valid()
