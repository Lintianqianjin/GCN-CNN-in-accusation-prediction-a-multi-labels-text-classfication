import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D,Activation,\
                        Add,Flatten,Dense, Dropout, BatchNormalization,RepeatVector,\
                        Permute,Multiply,GRU, MaxPooling1D, Bidirectional,LSTM,SimpleRNN
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import time,datetime
import matplotlib.pyplot as plt
import pandas as pd
import time
from keras.models import load_model
import os

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# 定义超参数开始
X_dim = 400
Y_dim = 202
# 定义超参数结束

# CNN
def buildCNN():
    data_input = Input(shape=[X_dim])
    word_vec = Embedding(input_dim=80000 + 1,  # 80000个词+全零
                         input_length=400,  # 每个句子400词
                         output_dim=128,  # 输出一个词128维
                         mask_zero=0,
                         name='Embedding')(data_input)
    x = word_vec
    # 卷积核大小设置为3
    x = Conv1D(filters=512, kernel_size=[3], strides=1, padding='same', activation='relu')(x)
    x = GlobalMaxPool1D()(x)
    x = BatchNormalization()(x)
    x = Dense(1000, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(202, activation="sigmoid")(x)
    model = Model(inputs=data_input, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    # curT = datetime.datetime.utcfromtimestamp(time.time()).strftime("%d-%H-%M-%S")
    plot_model(model, to_file=f'../data/Model_Picture/CNN.png', show_shapes=True)
    return model


#CNN+Resnet
def block(x, kernel_size):
    x_Conv_1 = Conv1D(filters=128, kernel_size=[kernel_size], strides=1, padding='same')(x)
    x_Conv_1 = Activation(activation='relu')(x_Conv_1)
    # x_Conv_2 = Conv1D(filters=128, kernel_size=[kernel_size], strides=1, padding='same')(x_Conv_1)
    x_Conv_2 = Add()([x, x_Conv_1])
    x = Activation(activation='relu')(x_Conv_2)
    return x

def buildCNN_plus_Resnet():

    data_input = Input(shape=[X_dim])
    word_vec = Embedding(input_dim=80000 + 1,
                         input_length=400,
                         output_dim=128,
                         mask_zero=0,
                         name='Embedding')(data_input)

    block1 = block(x=word_vec, kernel_size=3)
    block2 = block(x=block1, kernel_size=3)
    block3 = block(x=block2, kernel_size=3)
    x = GlobalMaxPool1D()(block3)
    x = BatchNormalization()(x)
    x = Dense(1000, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(202, activation="sigmoid")(x)
    model = Model(inputs=data_input, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file=f'../data/Model_Picture/RESNET2.png', show_shapes=True)
    return model


# CNN+Attention
def attention(input=None, depth=None):
    # sample_dimension 256 输入（batch_size,400,256）
    # 输出（batch_size,400,1） 每一个样本的每一个词上只有一个值
    # （这就理解为这个词的权重）
    # tanh 激活到-1到1之间
    attention = Dense(1, activation='tanh')(input)
    # 变为（batch_size,400）
    attention = Flatten()(attention)
    # 分配权重
    attention = Activation('softmax')(attention)
    # 重复Depth次，shape变为[batch_size,depth(256)，400]
    attention = RepeatVector(depth)(attention)
    # 将depth作为第一维，shape变为[batch_size,400,256]
    attention = Permute([2, 1], name='attention_vec')(attention)
    # input [batch_size,400,sample_dimension(256)]
    # 各个值*对应的权重
    attention_mul = Multiply(name='attention_mul')([input, attention])
    return attention_mul

def buildCNN_plus_Attention():
    data_input = Input(shape=[X_dim])
    word_vec = Embedding(input_dim=80000 + 1,
                         input_length=400,
                         output_dim=128,
                         mask_zero=0,
                         name='Embedding')(data_input)
    x = word_vec
    x = Conv1D(filters=256, kernel_size=[3], strides=1, padding='same', activation='relu')(x)
    x = attention(input=x, depth=256)
    x = GlobalMaxPool1D()(x)
    x = BatchNormalization()(x)
    x = Dense(1000, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(202, activation="sigmoid")(x)
    model = Model(inputs=data_input, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file=f'../data/Model_Picture/CNNAtt.png', show_shapes=True)

    return model


# Bi-LSTM
def buildBiLSTM():
    data_input = Input(shape=[X_dim])
    word_vec = Embedding(input_dim=80000 + 1,  # 80000个词+全零
                         input_length=400,  # 每个句子400词
                         output_dim=128,  # 输出一个词128维
                         mask_zero=0,
                         name='Embedding')(data_input)
    x = word_vec
    # 卷积核大小设置为3
    x = Bidirectional(LSTM(8,return_sequences=False),merge_mode='concat')(x)
    # x = LSTM(16)(x)
    # x = GlobalMaxPool1D()(x)
    # x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(202, activation="sigmoid")(x)
    model = Model(inputs=data_input, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    # curT = datetime.datetime.utcfromtimestamp(time.time()).strftime("%d-%H-%M-%S")
    plot_model(model, to_file=f'../data/Model_Picture/BiLSTM.png', show_shapes=True)
    return model

# RNN
def buildSimpleRNN():
    data_input = Input(shape=[X_dim])
    word_vec = Embedding(input_dim=80000 + 1,  # 80000个词+全零
                         input_length=400,  # 每个句子400词
                         output_dim=128,  # 输出一个词128维
                         mask_zero=0,
                         name='Embedding')(data_input)
    x = word_vec
    # x = Flatten()(x)
    # 卷积核大小设置为3
    # x = LSTM(8,return_sequences=False,activation='relu')(x)
    x =  SimpleRNN(256,return_sequences=False,activation='relu')(x)
    # x = GlobalMaxPool1D()(x)
    # x = Flatten()(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)
    x = Dense(1000, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(202, activation="sigmoid")(x)
    model = Model(inputs=data_input, outputs=x)
    adam = Adam(lr=0.1)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # curT = datetime.datetime.utcfromtimestamp(time.time()).strftime("%d-%H-%M-%S")
    plot_model(model, to_file=f'../data/Model_Picture/RNN.png', show_shapes=True)
    return model

# 验证
def strictAccuracy(Y_pred,Y_real,prob_baseline = 0.5):

    all_right = 0
    for i in range(len(Y_pred)):
        cur_pred = np.where(Y_pred[i] > prob_baseline, int(1), int(0))
        if list(cur_pred) == list(Y_real[i]):
            all_right += 1
    return all_right / len(Y_real)

def top1Accuracy(Y_pred,Y_real):

    all_right = 0
    for i in range(len(Y_pred)):
        max_indexes = np.where(Y_pred[i]==np.max(Y_pred[i]))
        # 以防最大值同值，使用for else循环
        for max_index in max_indexes[0]:
            if Y_real[i][max_index] != 1:
                break
        else:
            all_right +=1
    return all_right / len(Y_real)

def valid():
    X_val = np.load('../data/FinalData/final_X_valid.npy')
    Y_val = np.load('../data/FinalData/final_Y_Valid.npy')

    modelFiles = os.listdir('trainedModels')
    accs = {'strictAccuracy':{},'top1Accuracy':{}}
    for modelName in modelFiles:
        model = load_model(os.path.join('trainedModels',modelName))
        print(f'{modelName} prediction start **********')

        Y_pred = model.predict(X_val)

        strict_acc = strictAccuracy(Y_pred,Y_val,prob_baseline=0.7)
        accs['strictAccuracy'][modelName.split('_')[0]] = strict_acc

        top1_acc = top1Accuracy(Y_pred,Y_val)
        accs['top1Accuracy'][modelName.split('_')[0]] = top1_acc

        print(f'{modelName} prediction end **********')

    print(accs)

def predict():
    pass

def X_train_Y_train_generator(X_train,Y_train,range_num = 555):
    for Batch_begin_index in range(range_num):
        # if Batch_begin_index + 512 > 284584:
        if Batch_begin_index == range_num-1:
            end_index = -1
        else:
            # end_index = Batch_begin_index + 512
            end_index = Batch_begin_index*512 + 512

        will_return = (Batch_begin_index,X_train[Batch_begin_index*512:end_index],Y_train[Batch_begin_index*512:end_index])
        yield will_return


def mainTrain():
    X_train = np.load('../data/FinalData/final_X_train.npy')
    X_test = np.load('../data/FinalData/final_X_test.npy')
    Y_train = np.load('../data/FinalData/final_Y_train.npy')
    Y_test = np.load('../data/FinalData/final_Y_test.npy')

    # CNN_model = buildCNN()
    CNN_plus_RNN = buildCNN_plus_Resnet()
    # BiLSTM = buildBiLSTM()
    # CNN_plus_ATT = buildCNN_plus_Attention()
    # rnn = buildSimpleRNN()

    test_accs = []
    cur_best_acc = 0

    for epoch_index in range(1, 11):
        for batch_id, X_train_batch, Y_train_batch in X_train_Y_train_generator(X_train, Y_train):
            print(f'epoch {epoch_index} batch {batch_id}')
            # print(X_train_batch.shape)
            CNN_plus_RNN.fit(x=X_train_batch, y=Y_train_batch, batch_size=512, epochs=1, verbose=1)

            if batch_id != 0 and batch_id % 50 == 0:
                all_right = 0
                # any_right = 0
                # for test_batch_id, X_test_batch, Y_test_batch in X_train_Y_train_generator(X_test, Y_test,range_num=20):
                print('predict start')
                # print(X_test.shape)
                Y_pred = CNN_plus_RNN.predict(X_test)
                print('predict finish')
                for i in range(len(Y_pred)):
                    # print(Y_test[i])
                    # print(Y_batch_pred)
                    # exit()
                    # print('compare start')
                    cur_pred = np.nan_to_num(Y_pred[i])
                    cur_pred = np.where(cur_pred > 0.5, int(1), int(0))
                    # print('compare end')
                    if list(cur_pred) == list(Y_test[i]):
                        all_right += 1

                cur_acc = all_right / len(Y_test)

                print(f'当前测试集准确率{cur_acc}')
                # exit()
                test_accs.append(cur_acc)

                if cur_acc > 0.8 and cur_acc > cur_best_acc:
                    # todo:更换模型时要改名字
                    CNN_plus_RNN.save(f'trainedModels\\RES2NN_epoch_{epoch_index}_batch_{batch_id}_acc_{cur_acc}.h5')
                    cur_best_acc = cur_acc

    with open('RES2NN_acc.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_accs))
    # 画图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot([i for i in range(1, len(test_accs) + 1)], test_accs, label=u'测试集准确率')
    plt.legend()
    # todo: 更换模型时要改名字
    plt.savefig('RES2NN_acc2.png', dpi=300)


def committeeVote():
    X_val = np.load('../data/FinalData/final_X_valid.npy')
    Y_val = np.load('../data/FinalData/final_Y_Valid.npy')

    modelFiles = os.listdir('trainedModels')
    accs = {'strictAccuracy': 0, 'top1Accuracy': 0}
    models = []
    for modelName in modelFiles:
        model = load_model(os.path.join('trainedModels', modelName))
        models.append(model)

    print(f'{modelName} prediction start **********')

    Y_preds = [model.predict(X_val) for model in models]

    new_Y_pred = []

    for i in range(len(Y_val)):
        cur_pred = 0
        for each_preds in Y_preds:
            cur_pred += each_preds[i]

        cur_pred = np.where(cur_pred>=2,int(1),int(0))
        # print(cur_pred)

        new_Y_pred.append(cur_pred)

    new_Y_pred = np.array(new_Y_pred)


    strict_acc = strictAccuracy(new_Y_pred, Y_val, prob_baseline=0.625)
    accs['strictAccuracy'] = strict_acc

    top1_acc = top1Accuracy(new_Y_pred, Y_val)
    accs['top1Accuracy'] = top1_acc

    print(f'{modelName} prediction end **********')

    print(accs)

if __name__ == '__main__':
    valid()