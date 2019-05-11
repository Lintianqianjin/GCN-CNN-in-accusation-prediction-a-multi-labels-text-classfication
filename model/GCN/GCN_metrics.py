import tensorflow as tf
import numpy as np

def masked_sigmoid_cross_entropy(preds, labels, mask):
    def np_mask(preds,labels, mask):
        labels = np.array(labels,dtype=np.float32)
        preds = np.array(preds,dtype=np.float32)
        return preds[mask],labels[mask]
    masked_pred,masked_labels = tf.py_func(np_mask, [preds, labels,mask], [tf.float32,tf.float32])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=masked_pred, labels=masked_labels)
    # mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    # loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    one = tf.ones_like(preds)
    zero = tf.zeros_like(preds)
    onehotPred = tf.where(preds <0.5, x=zero, y=one)

    def np_mask_acc(onehotPred,labels, mask):
        onehotPred = np.array(onehotPred,dtype=np.float32)
        labels = np.array(labels,dtype=np.float32)
        accList = np.array([1 for pred,true in zip(onehotPred[mask],labels[mask])
                            if (pred==true).all()],dtype=np.float32)
        return accList

    correct_prediction = tf.py_func(np_mask_acc,[onehotPred,labels,mask],tf.float32)
    # tf.print(correct_prediction)
    return tf.reduce_mean(correct_prediction)

if __name__ == '__main__':
    # a = np.array([[0,0,0,0,1],
    #               [1,1,0,0,0],
    #               [0,1,0,0,1]])
    # b = np.array([[0,1,0,0,1],
    #               [1,1,0,0,0],
    #               [0,1,0,0,0]])
    # c= tf.equal(a,b)
    # sess = tf.Session()
    # op = sess.run(c)
    # print(op)
    # b = [False,False,True]
    # print(a[b])
    # mask1 = tf.constant([False,False,True,True,False])
    #
    # mask2 = tf.cast(mask1, dtype=tf.float32)
    # mask2 /= tf.reduce_mean(mask2)
    #
    # sess = tf.Session()
    # a = sess.run(mask1)
    # print(a)
    # a = sess.run(mask2)
    # print(a)
    # a = sess.run(mask3)
    # print(a)

    pass