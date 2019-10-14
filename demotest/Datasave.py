# import tensorflow as tf
# from numpy.random import RandomState
#
# bate_size = 8
# w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1),name='w1')
# w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1),name='w2')
#
# x = tf.placeholder(tf.float32,shape=[None,2],name='x-input')
# y_ = tf.placeholder(tf.float32,shape=[None,1],name='y-iutput')
#
# a = tf.matmul(x,w1)
# y = tf.matmul(a,w2)
#
# cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
# train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#
# ram = RandomState(1)
# dataset_set = 128
# X = ram.rand(dataset_set,2)#生成随机的模拟数据集
# Y = [[int(x1+x1<1)] for x1,x2 in X]
#
# with tf.Session() as sess:
#     init_op = tf.initialize_all_variables()
#     sess.run(init_op)
#     print(sess.run(w1))
#     print(sess.run(w2))
#
#     # w1 = [[-0.81131822,1.48459876,0.06532937],[-2.44270396,0.0992484,0.59122431]]
#     # w2 = [[-0.81131822],[1.48459876],[0.06532937]]
#
#     steps = 5000
#     for i in range(steps):
#         start = (i*bate_size)%dataset_set
#         end = min(start+bate_size,dataset_set)
#
#         sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
#         if i%1000 == 0:
#             total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
#             print("After %d training step(s),cross entropy on all data is %g"%(i,total_cross_entropy))
#     print(sess.run(w1))
#     print(sess.run(w2))
# writer = tf.summary.FileWriter('E:\GitHub\All-excecise\logs',tf.get_default_graph())
# writer.close()
#
#
import pandas as pd
import numpy as np

testfile = '0.822sub_0.csv'
testresult = pd.read_csv(testfile)
testarray = np.array(testresult[['label_0','label_1','label_2']])
dataid = testresult[['id']]
datalen = len(testresult)
testlist = testarray.tolist()
datadict = {'label':[]}
for i in range(datalen):
    index = testlist[i].index(max(testlist[i]))
    datadict['label'].append(index)
myresult = pd.DataFrame(datadict)
result = pd.concat([dataid,myresult],axis=1)
result.to_csv('robera_large_0.csv',index=False)




