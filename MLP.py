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
n_hidden_1 = 25
n_class = y_train.shape[1]

x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_class])


def multilayer_perceptron(x, weight, bias):
    layer1 = tf.add(tf.matmul(x, weight['h1']), bias['h1'])
    layer1 = tf.nn.relu(layer1)
    out_layer = tf.add(tf.matmul(layer1, weight['out']), bias['out'])

    return out_layer

weight = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_class]))
}
bias = {
    'h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_class]))
}

# 建立模型
pred = multilayer_perceptron(x, weight, bias)

# 定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

# 优化
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.initialize_all_variables()

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# 训练模型
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_sample / batch_size)

        for i in range(total_batch):
            _, c = sess.run([optimizer, cost], feed_dict={x: X_train[i*batch_size: (i+1)*batch_size, :],
                                                          y: y_train[i*batch_size: (i+1)*batch_size, :]})
            avg_cost += c / total_batch

        plt.plot(epoch+1, avg_cost, 'co')

        if epoch % display_step == 0:
            print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))

    print('Optimization Finished!')

    # Test
    acc = accuracy.eval({x: X_test, y: y_test})
    print('Accuracy:', acc)

    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('lr=%f, te=%d, bs=%d, acc=%f' % (learning_rate, training_epochs, batch_size, acc))
    plt.tight_layout()
    plt.savefig('XXXX.png', dpi=200)

    plt.show()
