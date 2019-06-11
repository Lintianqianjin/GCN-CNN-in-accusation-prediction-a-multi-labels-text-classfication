from GCN_init import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 support=None, sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        if support is None:
            self.support = placeholders['support'][0]
        else:
            self.support = support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(1):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        # supports = list()
        # for i in range(len(self.support)):
        #     if not self.featureless:
        #         pre_sup = dot(x, self.vars['weights_' + str(i)],
        #                       sparse=self.sparse_inputs)
        #     else:
        #         pre_sup = self.vars['weights_' + str(i)]
        #     support = dot(self.support[i], pre_sup, sparse=True)
        #     supports.append(support)
        # output = tf.add_n(supports)
        if not self.featureless:
            pre_sup = dot(x, self.vars['weights_0'],
                          sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights_0']
        output = dot(self.support, pre_sup, sparse=True)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

#
# ###keras 开始###
# # 该层接受的输入应该是两个INPUT类，分别是邻接矩阵和特征矩阵
# class gcnlayer(Layer):
#
#     def __init__(self, output_dim,
#                  activation=None,
#                  # 偏置项，即常数项
#                  use_bias=True,
#                  # 核矩阵，对输入X的变化矩阵，权矩阵
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  # 施加在权重上的正则项
#                  kernel_regularizer=None,
#                  # 施加在偏置向量上的正则项
#                  bias_regularizer=None,
#                  # 施加在输出上的正则项
#                  activity_regularizer=None,
#                  # 对主权重矩阵进行约束
#                  kernel_constraint=None,
#                  # 对偏置向量进行约束
#                  bias_constraint=None,
#                  **kwargs):
#         # 初始化各参数
#         self.output_dim = output_dim
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.activity_regularizer = regularizers.get(activity_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)
#         self.supports_masking = True
#
#
#         super(gcnlayer, self).__init__(**kwargs)
#         pass
#
#     # todo: 定义该层需要训练的权重矩阵
#     def build(self, input_shape):
#         # 0是选中特征矩阵
#         input_feature_dim = input_shape[0][1]
#
#         # 定义卷积核
#         self.kernel = self.add_weight(shape=(input_feature_dim, self.output_dim),
#                                       initializer=self.kernel_initializer,
#                                       name='kernel',
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint)
#
#         # todo: 暂时没有考虑常数项
#
#         super(gcnlayer, self).build(input_shape)
#
#     # todo: 定义该层完成的变换过程
#     def call(self, inputs, **kwargs):
#         features_matrix = inputs[0]
#         adj_matrix = inputs[1]
#
#         # 特征矩阵与邻接矩阵做积，得到接受了邻接节点信息后的新的节点特征
#         propagated_feature_matrix = K.dot(adj_matrix, features_matrix)
#
#         # 接下来做卷积转化为新的特征表示
#         output = K.dot(propagated_feature_matrix, self.kernel)
#
#         # todo:暂时没有考虑常数项
#
#         # 激活，如果是隐藏层用relu,最后一层softmax或sigmoid
#         return self.activation(output)
#
#
#     # todo: 定义该层输入输出的形状变化
#     def compute_output_shape(self, input_shape):
#         # 输出为（节点数*新的特征数)
#         # 新的特征数也就是该层神经元数，即kernel的宽
#
#         features_matrix = input_shape[0]
#         output_shape = (features_matrix[0], self.output_dim)
#         return output_shape
#
#     # todo: 返回该层的各参数
#     def get_config(self):
#         pass
#
# ###keras 结束###
