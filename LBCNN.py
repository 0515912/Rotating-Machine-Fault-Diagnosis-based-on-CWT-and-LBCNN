# Local Binary Convolution Neural Network
# Author: Matthew Lin
# Date: 31/03/2019

from keras.layers.core import Layer
from keras.engine import InputSpec
from keras import backend as K
from keras import initializers
from keras.layers.convolutional import Conv2D, _Conv
from keras import initializers, constraints, activations, regularizers
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations
from scipy.stats import bernoulli
import numpy as np
from keras.utils import conv_utils
import tensorflow as tf
def get_anchor_weight(kernel_size, in_channel, out_channel, sparsity):
    # the number of anchor weights
    num = kernel_size[0]*kernel_size[1]*in_channel*out_channel
    weights = np.zeros((num, 1))
    # the number of non-zero weights
    max_num = int(sparsity*num+0.5)
    # distribute the weights randomly
    index = np.arange(num)
    np.random.shuffle(index)
    index = index[0:max_num]
    for i in index:
        weights[i] = bernoulli.rvs(0.5)*2 - 1
    weights = np.reshape(weights, [kernel_size[0], kernel_size[1], in_channel, out_channel])
    return weights

class LBC(Conv2D):
    def __init__(self, filters,
                 kernel_size,
                 rank=2,
                 sparsity=0.9,
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable = False,
                 **kwargs):
        self.rank = rank
        self.sparsity = sparsity
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank,
                                                      'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.trainable = trainable
        super(LBC, self).__init__(filters=filters, kernel_size=kernel_size, **kwargs)


    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        #self.kernel = get_anchor_weight(self.kernel_size, input_shape[-1], self.filters, self.sparsity)
        self.kernel = tf.Variable(get_anchor_weight(self.kernel_size, input_shape[-1], self.filters, self.sparsity).astype(np.float32), trainable=False)
        self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
        super(LBC, self).build(input_shape)  # Be sure to call this at the end

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.get(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(_Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))