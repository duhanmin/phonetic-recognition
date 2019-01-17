#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Time    : 2017/12/29 14:05
# @Author  : 杜函敏
# @Email   : duhanmin@foxmail.com
# @File    : 语音识别.py
# @Software: PyCharm Community Edition

import tensorflow as tf
import scipy.io.wavfile as wav
from python_speech_features import mfcc,delta
import os
import numpy as np
import sklearn.preprocessing

path = r"data/xunlian/"

test_path = r"data/test_data/"
isnot_test_path = r"data/isnot_test_path/"

#使用one-hot编码，将离散特征的取值扩展到了欧式空间
#全局one-hot编码空间
label_binarizer = ""
def def_one_hot(x):
    if label_binarizer == "":
        binarizer = sklearn.preprocessing.LabelBinarizer()
    else:
        binarizer = label_binarizer
    binarizer.fit(range(max(x)+1))
    y= binarizer.transform(x)
    return y

def read_wav_path(path):

    map_path, map_relative = [str(path) + str(x) for x in os.listdir(path) if os.path.isfile(str(path) + str(x))], [y for y in os.listdir(path)]
    return map_path, map_relative

def def_wav_read_mfcc(file_name):
    fs, audio = wav.read(file_name)
    processed_audio = mfcc(audio, samplerate=fs, nfft=2000)
    return processed_audio

def find_matrix_max_shape(audio):
    h, l = 0, 0
    for a in audio:
        a, b = np.array(a).shape
        if a > h:
            h=a
        if b > l:
            l=b
    return 700, 13

def matrix_make_up(audio):
    h, l = find_matrix_max_shape(audio)
    print(h,l)
    print(type(h))
    new_audio = []
    for aa in audio:
        zeros_matrix = np.zeros([h, l],np.int8)
        a, b = np.array(aa).shape
        for i in range(a):
            for j in range(b):
                zeros_matrix[i, j]=zeros_matrix[i,j]+aa[i,j]
        new_audio.append(zeros_matrix)
    return new_audio,h,l

def read_wav_matrix(path):
    map_path, map_relative = read_wav_path(path)
    audio=[]
    labels=[]
    for idx, folder in enumerate(map_path):
        processed_audio_delta = def_wav_read_mfcc(folder)
        audio.append(processed_audio_delta)
        labels.append(int(map_relative[idx].split(".")[0].split("_")[0]))
    x_data,h,l = matrix_make_up(audio)
    x_data = np.array(x_data)
    x_label = np.array(def_one_hot(labels))
    return x_data, x_label, h, l

#初始化权值
def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev=0.01)#生成一个截断的正态分布
    return tf.Variable(initial,name=name)

#初始化偏置
def bias_variable(shape,name):
    initial = tf.constant(0.01,shape=shape)
    return tf.Variable(initial,name=name)

#卷积层
def conv2d(x,W):
    #x input tensor of shape `[batch, in_height, in_width, in_channels]`
    #W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    #`strides[0] = strides[3] = 1`. strides[1]代表x方向的步长，strides[2]代表y方向的步长
    #padding: A `string` from: `"SAME", "VALID"`
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化层
def max_pool_2x2(x):
    #ksize [1,x,y,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def xunlianlo(path,test_path):
    x_train, y_train, h, l = read_wav_matrix(path)
    x_test, y_test, h, l = read_wav_matrix(test_path)

    print(x_train.shape)

    m,n = y_train.shape
    # 命名空间
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, h, l], name='x-input')
    y = tf.placeholder(tf.float32, [None, n], name='y-input')
    # 改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]`
    x_image = tf.reshape(x, [-1, h, l, 1], name='x_image')

    # 初始化第一个卷积层的权值和偏置
    W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')  # 5*5的采样窗口，32个卷积核从3个平面抽取特征
    b_conv1 = bias_variable([32], name='b_conv1')  # 每一个卷积核一个偏置值

    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    h_conv1 = tf.nn.leaky_relu(conv2d_1)
    h_pool1 = max_pool_2x2(h_conv1)  # 进行max-pooling

    # 初始化第二个卷积层的权值和偏置
    W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')  # 5*5的采样窗口，64个卷积核从32个平面抽取特征
    b_conv2 = bias_variable([64], name='b_conv2')  # 每一个卷积核一个偏置值

    # 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数

    conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    h_conv2 = tf.nn.leaky_relu(conv2d_2)
    h_pool2 = max_pool_2x2(h_conv2)  # 进行max-pooling

    # 300*300的图片第一次卷积后还是300*300，第一次池化后变为150*150
    # 第二次卷积后为150*150，第二次池化后变为了75*75
    # 进过上面操作后得到64张7*7的平面

    # 初始化第一个全连接层的权值
    W_fc1 = weight_variable([175 * 4  * 64, 1024], name='W_fc1')  # 上一场有75*75*64个神经元，全连接层有1024个神经元
    b_fc1 = bias_variable([1024], name='b_fc1')  # 1024个节点

    # 把池化层2的输出扁平化为1维
    h_pool2_flat = tf.reshape(h_pool2, [-1, 175 * 4 * 64], name='h_pool2_flat')

    # 求第一个全连接层的输出
    wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc1 = tf.nn.leaky_relu(wx_plus_b1)

    # keep_prob用来表示神经元的输出概率
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

    # 初始化第二个全连接层
    W_fc2 = weight_variable([1024, n], name='W_fc2')
    b_fc2 = bias_variable([n], name='b_fc2')
    wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # 计算输出
    prediction = tf.nn.leaky_relu(wx_plus_b2)

    tf.add_to_collection('predictions', prediction)

    p = tf.nn.softmax(wx_plus_b2)

    tf.add_to_collection('p', p)
    # 交叉熵代价函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                   name='cross_entropy')

    # 使用AdamOptimizer进行优化
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(5).minimize(cross_entropy)

    # 求准确率
    # 结果存放在一个布尔列表中
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # argmax返回一维张量中最大的值所在的位置
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #保存模型使用环境
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 创建一个协调器，管理线程
        coord = tf.train.Coordinator()
        # 启动QueueRunner, 此时文件名队列已经进队
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(100001):
            # 训练模型
            sess.run(train_step, feed_dict={x: x_train, y: y_train, keep_prob: 1.0})

            test_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: x_train, y: y_train, keep_prob: 1.0})
            print("训练第 " + str(i) + " 次, 训练集准确率= " + str(train_acc) + " , 测试集准确率= " + str(test_acc))

            if test_acc == 1 and train_acc >= 0.95:
                print("准确率完爆了")
                # 保存模型
                saver.save(sess, 'nn/my_net.ckpt')
                break

        # 通知其他线程关闭
        coord.request_stop()
        # 其他所有线程关闭之后，这一函数才能返回
        coord.join(threads)
def test_main(isnot_test_path):
    # 本地情况下生成数据
    x_test, y_test, h, l = read_wav_matrix(isnot_test_path)
    m,n = y_test.shape

    # 迭代网络
    with tf.Session() as sess:
        # 保存模型使用环境
        saver = tf.train.import_meta_graph("nn/my_net.ckpt.meta")
        saver.restore(sess, 'nn/my_net.ckpt')

        predictions = tf.get_collection('predictions')[0]
        p = tf.get_collection('p')[0]

        graph = tf.get_default_graph()

        input_x = graph.get_operation_by_name('x-input').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

        for i in range(m):
            result = sess.run(predictions, feed_dict={input_x: np.array([x_test[i]]),keep_prob:1.0})
            haha = sess.run(p, feed_dict={input_x: np.array([x_test[i]]), keep_prob: 1.0})
            print("取值置信度"+str(haha))

            print("实际 :"+str(np.argmax(y_test[i]))+" ,预测: "+str(np.argmax(result))+" ,预测可靠度: "+str(np.max(haha)))

if __name__ == '__main__':

    xunlianlo(path,test_path)

    # test_main(isnot_test_path)
