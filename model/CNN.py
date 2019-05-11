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



def get_session(gpu_fraction=0.6):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))




def crime2onehot(crime,labels):
    test = np.array(crime)
    labels = np.array(labels)
    mask1 = np.in1d(labels,test)
    mask0 = np.in1d(labels,test,invert=True)
    labels[mask0] = 0
    labels[mask1] = 1
    return labels


def generate_train_and_labels():
    small = open(r'../data/小数据集.json','r',encoding='utf-8')
    crime_label = [0, 2, 3, 8, 12, 16, 11, 27]

    nd_train = []
    nd_label = []

    for index,line in enumerate(small):
        print(index)
        cur = json.loads(line.strip())
        nd_train.append([int(word) for word in cur['fact']])
        one_hot = crime2onehot(crime=cur['accusation'],labels=crime_label)
        nd_label.append(one_hot)

    nd_train = np.array(nd_train)
    nd_label = np.array(nd_label)

    np.save(r'../data/小训练集.npy',nd_train)
    np.save(r'../data/小标签集.npy', nd_label)

def predict2top(predictions):
    one_hots = []

    for prediction in predictions:
        max = prediction.max()
        one_hot = np.where(prediction == max , int(1), int(0))
        one_hots.append(one_hot)
    return np.array(one_hots)

if __name__ == '__main__':
    #KTF.set_session(get_session())
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    #generate_train_and_labels()
    #c = np.load("../data/小训练集.npy")
    #print(c[0][0:10])
    #训练集
    fact = np.load('../data/小训练集.npy')
    fact_train, fact_test = train_test_split(fact, test_size=0.05, random_state=1)
    del fact

    # 标签数据集
    labels = np.load('../data/小标签集.npy')
    labels_train, labels_test = train_test_split(labels, test_size=0.05, random_state=1)
    del labels


    print([fact_train.shape[1]])
    data_input = Input(shape=[fact_train.shape[1]])
    word_vec = Embedding(input_dim=80000 + 1,   #80000个词+全零
                         input_length=400,      #每个句子400词
                         output_dim=128,        #输出一个词128维
                         mask_zero=0,
                         name='Embedding')(data_input)
    x = word_vec
    #卷积核大小设置为3
    x = Conv1D(filters=512, kernel_size=[3], strides=1, padding='same', activation='relu')(x)
    x = GlobalMaxPool1D()(x)
    x = BatchNormalization()(x)
    x = Dense(1000, activation="relu")(x)
    x = Dropout(0.2)(x)
    print(labels_train.shape[1])
    x = Dense(labels_train.shape[1], activation="sigmoid")(x)
    model = Model(inputs=data_input, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    #batch_size一批多少样本
    #epoch 总样本滚多少次

    for index in range(1, 21):
        print(f'epoch  {index}')
        model.fit(x=fact_train, y=labels_train, batch_size=256, epochs=1, verbose=1)

        print(fact_test[:10])

        y = model.predict(fact_test[:])
        y1 = predict2top(y)
        #y2 = predict2half(y)
        #y3 = predict2both(y)
        right = 0
        for i in range(len(y1)):
            #a = labels_test_
            #b = y1[i]
            if list(labels_test[i]) == list(y1[i]):
                right+=1
        print('当前测试集准确率'+str(right / len(y1)))
        model.save('models_trained\\CNN_train_3_epochs_' + str(index)+'_accuracy_'+str(right / len(y1)) + '.h5')
