"""
Created by Li Jinku in 2017/4/25
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import csv
import tensorflow as tf
import numpy as np
import pandas as pd


# 程序开始时间
startTime = time.clock()

# 提取训练数据
trainData = pd.read_csv('./data/train.csv')
trainSet = trainData.loc[0 : ][['V1', 'V2', 'V3', 'V4', 'L', 'Alpha', 'Ka', 'Kc', 'KsModified']]
trainSet = np.array(trainSet)
trainTargets = trainData.loc[0 : ]['Vt']
trainTargets = np.array(trainTargets) - np.array([[0.0]])
trainTargets = np.transpose(trainTargets)

# 提取测试数据
testData = pd.read_csv('./data/test.csv')
testSet = testData.loc[0 : ][['V1', 'V2', 'V3', 'V4', 'L', 'Alpha', 'Ka', 'Kc', 'KsModified']]
testSet = np.array(testSet)
testTargets = testData.loc[0 : ]['Vt']
testTargets = np.array(testTargets) - np.array([[0.0]])
testTargets = np.transpose(testTargets)

print('Reading Data Finished!')


# 设置模型参数
learningRate = 0.005
trainingEpochs = 500
batchSize = 500
displayStep = 1
sampleNum = trainSet.shape[0]
inputNum = trainSet.shape[1]
outputNum = 1
hidden1Num = 20

x = tf.placeholder(tf.float32, [None, inputNum])
y = tf.placeholder(tf.float32, [None, outputNum])

weights = {
    'h1': tf.Variable(tf.random_normal([inputNum, hidden1Num])),
    'out': tf.Variable(tf.random_normal([hidden1Num, outputNum]))
}
biases = {
    'h1': tf.Variable(tf.random_normal([hidden1Num])),
    'out': tf.Variable(tf.random_normal([outputNum]))
}

def multilayerPerceptron(x, weight, bias):
    layer1 = tf.add(tf.matmul(x, weight['h1']), bias['h1'])
    layer1 = tf.nn.sigmoid(layer1)
    out_layer = tf.add(tf.matmul(layer1, weight['out']), bias['out'])

    return out_layer

# 建立模型
pred = multilayerPerceptron(x, weights, biases)

# 定义损失函数 cost or loss
loss = tf.reduce_mean(tf.square(pred - y))

# 优化算子 optimizer or trainStep
trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

# 初始化所有变量
init = tf.global_variables_initializer()

relativeErrorMean = tf.reduce_mean((pred - y) / y)


# 训练模型
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(trainingEpochs):
        avgCost = 0
        totalBatch = int(sampleNum / batchSize)

        for i in range(totalBatch):
            _, c = sess.run([trainStep, loss], feed_dict={x: trainSet[i * batchSize : (i + 1) * batchSize, :],
                                                          y: trainTargets[i * batchSize : (i + 1) * batchSize, :]})
            avgCost += c / totalBatch

        if epoch % displayStep == 0:
            trainRelativeErrorMean = relativeErrorMean.eval(feed_dict={x: trainSet, y: trainTargets})
            print('Epoch:', '%03d' % (epoch + 1), 'Cost =', '{:.9f}'.format(avgCost), 'Training Mean Relative Error: %.3f%%' % (trainRelativeErrorMean * 100))

    print('Optimization Finished!\n')

    # 针对测试数据输出预测速度，非常重要！！！
    yPred = sess.run(pred, feed_dict={x: testSet}) # 这一句代码非常重要！！！
    testRelativeError = (yPred - testTargets) / testTargets
    testRelativeErrorMean = tf.reduce_mean(testRelativeError)

    # 得到output.csv文件
    if os.path.isfile('./data/output.csv'):
        os.remove('./data/output.csv')
    with open('./data/output.csv', 'w', newline='') as outFiles:
        fieldnames = ['Predicted Velocity', 'Test Relative Error']
        writer = csv.DictWriter(outFiles, fieldnames=fieldnames)
        writer.writeheader()
        for i in len(yPred):
            writer.writerow({'Predicted Velocity' : yPred[i], 'Test Relative Error' : testRelativeError[i]})
    print('"output.csv" is finished!')
    print('Testing Mean Relative Error: %.3f%%\n' % (testRelativeErrorMean * 100))


# 程序结束时间，最后输出整个程序运行的时间
endTime = time.clock()
print('The program elapsed time is', endTime - startTime, 's.')