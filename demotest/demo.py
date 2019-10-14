# import tensorflow as tf
# from numpy.random import RandomState
#
# bate_size = 8
# w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
# # w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#
# x = tf.placeholder(tf.float32,shape=[None,2],name='x-input')
# y_ = tf.placeholder(tf.float32,shape=[None,1],name='y-iutput')
#
# # a = tf.matmul(x,w1)
# y = tf.matmul(x,w1)
#
# loss_less = 10
# loss_more = 1
# loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*loss_less,(y_-y)*loss_more))
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
#
# ram = RandomState(1)
# dataset_set = 128
# X = ram.rand(dataset_set,2)#生成随机的模拟数据集
# Y = [[x1 + x2 + ram.rand()/10.0 - 0.05] for x1,x2 in X]
#
# with tf.Session() as sess:
#     init_op = tf.initialize_all_variables()
#     sess.run(init_op)
#     print(sess.run(w1))
#     # print(sess.run(w2))
#
#     # w1 = [[-0.81131822,1.48459876,0.06532937],[-2.44270396,0.0992484,0.59122431]]
#     # w2 = [[-0.81131822],[1.48459876],[0.06532937]]
#
#     steps = 50000
#     for i in range(steps):
#         start = (i*bate_size)%dataset_set
#         end = min(start+bate_size,dataset_set)
#
#         sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
#         if i%1000 == 0:
#             loss = sess.run(loss,feed_dict={x:X,y_:Y})
#             print("After %d training step(s),loss on all data is %g"%(i,loss))
#     print(sess.run(w1))
#     # print(sess.run(w2))
#
# import numpy as np
# import tensorflow as tf
# a= tf.constant([[1,2,3],[3,4,5],[5,6,7]],dtype=float)
# matric = [[1,2,3],[3,4,5],[5,6,7]]
# matric = np.mat(matric)
# b = a**2
# c = tf.sqrt(b)
# print(matric**2)
# with tf.Session() as sess:
#     print(tf.matmul(a,a).eval())
#     print(b.eval())
#     print(c.eval())

# testtuple = [(1,'a'),(2,'b'),(3,'c'),(4,'d')]
# print(list(zip(*testtuple)))

import numpy as np
import time
# arties = [['陈','志','妙','d','a','a','b'],['f','志','妙','s','a','d','b'],['陈','y','c','d','a','a','b']]
# capital = set()
# for i in range(len(arties)):
#     capital = capital | set(arties[i])
# print(sorted(capital))
# vectors = (np.ones(5000000))
# starttime = time.time()
# vectors = (map(lambda x:int(x),vectors))
# vectors = list(vectors)
# endtime = time.time()
#
#
# print(endtime-starttime)
# vectors1 = (np.ones(5000000))
# starttime = time.time()
# vectors1 = [int(vector) for vector in vectors1]
# endtime = time.time()
# print(endtime-starttime)
# import jieba
# import re
# import torch
#
# sentence = "中文分词研究背景以及意义，上海自来水来自海上，一般的词难不倒我，在有些句子中将出错。是这样吗？？！！"
# character =['，','。','！','？','；']
# jieba.suggest_freq(('中','将'),True)
# words = jieba.cut(sentence)
# word_list=[]
# for word in words:
#     if word not in character:
#         word_list.append(word)
# print(word_list)

import torch

states = ('Healthy', 'Fever')
observations = ('normal', 'cold', 'dizzy')
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
transition_probability = {'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
                        'Fever': {'Healthy': 0.4, 'Fever': 0.6},}
emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}
def print_dptable(V):
    print("    ")
    for i in range(len(V)): print("%7d" % i)
    for y in V[0].keys():
        print("%.5s: " % y)
        for t in range(len(V)):
            print("%.7s" % ("%f" % V[t][y]))

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
    # Run Viterbi for t > 0
    for t in range(1,len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            (prob, state) = max([(V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        # Don't need to remember the old paths
        path = newpath
    print_dptable(V)
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    return (prob, path[state])

def example():
    return viterbi(observations,states,start_probability,transition_probability,emission_probability)
print(example())

