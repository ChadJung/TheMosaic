import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import confusion_matrix


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,               # 입력으로 들어갈 레이어
                   num_input_channels,  # 입력 레이어의 depth
                   filter_size,         # 컨벌루션 필터 종횡 크기
                   num_filters,         # 필터 갯수(output depth)
                   use_pooling=True,
                   rate=0.1):   # pooling 여부
    shape = [filter_size, filter_size, num_input_channels, num_filters]     # weights에 들어갈 shape
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    # same padding(결과 크기 같음) 방식으로 하나도 빠짐없이(strides) convolution 함
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases
    layer = tf.nn.relu(layer)

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,             # 컨벌루션 한 레이어
                               ksize=[1, 2, 2, 1],     # kernel size(2x2)
                               strides=[1, 2, 2, 1],   # 1칸씩 건너뛰면서(batch depth 말고 w, h)
                               padding='SAME')          # same padding

    layer = tf.nn.dropout(layer, rate=rate)
    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


def new_fc_layer(input,         # fully connected 만들기 전 레이어
                 num_inputs,
                 num_outputs,
                 use_relu=True):  # Rectified Linear Unit 쓸지

    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


