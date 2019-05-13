# GCN CNN in accusation predict
## a multi-labels text classfication

#### 2019/5/13
##### add a GCN baseline based on tensorfow  
the model is composed of three graph convolutional layer and a dense layer  
the dataset contains 499 texts of criminal fact and involves 5 labels, which are 盗窃,故意伤害,诈骗,危险驾驶,抢劫(Theft, intentional injury, fraud, dangerous driving, robbery), and all the texts contains 8182 different words.  
However, as a result, the loss(sigmoid_cross_entropy) is nearly no longer reduced when it is not a very small value(0.6002855,after 200 epoch).  
The reason may be limited by the size of the data set(only 499 samples).  

·try 998 texts, 12491 words, 5 labels, out of memory·

###### todo: figure out sparseTensor multiplication to make it possible to operate on more larger matrices
