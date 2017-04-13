from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 读取数据-需要换！！！
data1 = unpickle('cifar-10-batches-py/data_batch_1')
data2 = unpickle('cifar-10-batches-py/data_batch_2')
data3 = unpickle('cifar-10-batches-py/data_batch_3')
data4 = unpickle('cifar-10-batches-py/data_batch_4')
data5 = unpickle('cifar-10-batches-py/data_batch_5')

X_train = np.concatenate((data1['data'], data2['data'], data3['data'], data4['data'], data5['data']), axis=0)
label = np.concatenate((data1['labels'], data2['labels'], data3['labels'], data4['labels'], data5['labels']), axis=0)
y_train = onehot(label)

test = unpickle('cifar-10-batches-py/test_batch')
X_test = test['data']
y_test = onehot(test['labels'])

# 设置模型参数
learning_rate = 0.01
training_epochs = 500
batch_size = 500
display_step = 1
n_sample = X_train.shape[0]

n_input = X_train.shape[0]
n_hidden1 = 25
n_class = y_train.shape[1]

x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_class])


def multilayer_perceptron(x, weight, bias):
    layer1 = tf.add(tf.matmul(x, weight['h1']), bias['h1'])
    layer1 = tf.nn.relu(layer1)
    out_layer = tf.add(tf.matmul(layer1, weight['out']), bias['out'])

    return out_layer

weight = {
    
}