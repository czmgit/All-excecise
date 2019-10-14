# import tensorflow as tf
#
# def get_weight(shape,lambd):
#     var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
#     tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambd)(var))
#     return var
#
# x = tf.placeholder(tf.float32,shape=(None,2))
# y_ = tf.placeholder(tf.float32,shape=(None,1))
#
# batch_size = 8
# layer_dimension = [2,10,10,10,1]
# n_layers = len(layer_dimension)
# cur_layer = x
# in_dimension = layer_dimension[0]
# for i in range(1,n_layers):
#     out_dimension = layer_dimension[i]
#     weight = get_weight([in_dimension,out_dimension],0.001)
#     bias = tf.Variable(tf.constant(0.1,shape=[out_dimension]))
#     cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
#     in_dimension = layer_dimension[i]
# mse_loss = tf.reduce_mean(tf.square(y_-cur_layer))
# tf.add_to_collection('losses',mse_loss)
# loss = tf.add_n(tf.get_collection('losses'))
# writer = tf.summary.FileWriter('E:\GitHub\All-excecise\logs',tf.get_default_graph())
# writer.close()
#
# # import tensorflow as tf
# # a = tf.constant([10.0,20.0,40.0],name='a')
# # b = tf.Variable(tf.random_uniform([3]),name='b')
# # output = tf.add_n([a,b],name='add')
# # writer = tf.summary.FileWriter('E:\GitHub\All-excecise\logs',tf.get_default_graph())
# # writer.close()

# import tensorflow as tf
# a = [[1,2,3],[4,5,6]]
# b = [[7,8,9],[0,1,2]]
# concattest = tf.concat([a,b],1)
# with tf.Session() as sess:
#     print(concattest.eval())
# L = [('a',3),('c',1),('d',2),('b',4)]
# temp = [(i,k)for i,k in L if k >=2]
# print(temp)
# dict1 = {'e':567,'f':123,'g':678,'h':345}
# temp = (sorted(dict1.items(),key=lambda x:(-x[1])))
# print(temp)
# num=lambda x:(x[1],x[0])
# list=[]
# for k,i in enumerate(dict1):
#     print(num((i,k)))
#     list.append(num((i,k)))
# print(sorted(list))

# import tensorflow as tf
# a = tf.constant([[[1,2,3],[1.1,2.2,3.3]],[[4,5,6],[4.1,5.1,6.1]],[[7,8,9],[7.1,8.1,9.1]],[[10,11,12],[10.1,11.1,12.1]]])
# with tf.Session() as sess:
#     print(a.eval(),'\n')
#     embedding = tf.nn.embedding_lookup(a,[0,1])
#     embedding1 = tf.nn.embedding_lookup(a,[1, 2,1])
#     temp = a**2
#     print(temp.eval())
#     print(embedding.eval(),'\n\n',embedding1.eval())
# import jieba
# seg_list = jieba.cut("你是这条gai最靓的仔",cut_all=False)
# print("/".join(seg_list))
# import random
# list = {'a':1,'b':2,'c':3,'d':4}
# temp = sorted(list.items())
# print(temp)
# print(type(temp))

import  os
import time
import random
import jieba,codecs
import nltk
import sklearn
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


def make_word_set(words_file):
    words_set = set()
    with codecs.open(words_file,'r','utf-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word)>0 and word not in words_set:
                words_set.add(word)
        return words_set

def text_processing(folder_path,test_size=0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    for folder in folder_list:
        new_folder_path = os.path.join(folder_path,folder)
        files = os.listdir(new_folder_path)
        j = 1
        for file in files:
            if j > 100:
                break
            with codecs.open(os.path.join(new_folder_path,file),'r','utf-8') as fp:
                raw = fp.read()
            # jieba.enable_parallel(4)#
            word_cut = jieba.cut(raw,cut_all=False)
            word_list = list(word_cut)
            # jieba.disable_parallel()
            data_list.append(word_list)
            class_list.append(folder)
            j += 1
    data_class_list = list(zip(data_list,class_list))
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*test_size)+1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list,train_class_list = list(zip(*train_list))
    test_data_list,test_class_list = list(zip(*test_list))
    #train_data_list,train_class_list,test_data_list,test_class_list = sklearn.cross_validation.train_test_split(data_list,class_list,test_size = test_size)
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word not in all_words_dict:
                all_words_dict[word] = 1
            else:
                all_words_dict[word] +=1
    all_words_tuple_list = sorted(all_words_dict.items(),key = lambda f:f[1],reverse = True)
    all_words_list = list(list(zip(*all_words_tuple_list))[0])

    return all_words_list,train_data_list,test_data_list,train_class_list,test_class_list

def words_dict(all_words_list,deleteN,stopwords_set = set()):
    feature_words = []
    n = 1
    for t in range(deleteN,len(all_words_list),1):
        if n > 2000:
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1<len(all_words_list[t])<5:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words

def text_features(train_data_list,test_data_list,feature_words,flag ='nltk'):
    def text_features(text,feature_words):
        text_words = set(text)
        if flag == 'nltk':
            features = {word:1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []
        return features
    train_feature_list = [text_features(text,feature_words)for text in train_data_list]
    test_feature_list = [text_features(text,feature_words)for text in test_data_list]
    return train_feature_list,test_feature_list

def text_classifier(train_feature_list,test_feature_list,train_class_list,test_class_list,flag ='nltk'):
    if flag =='nltk':
        train_flist = list(zip(train_feature_list,train_class_list))
        test_flist = list(zip(test_feature_list,test_class_list))
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        test_accuracy = nltk.classify.accuracy(classifier,test_flist)
    elif flag =='sklearn':
        classifier = MultinomialNB(alpha=0.38).fit(train_feature_list,train_class_list)
        test_accuracy = classifier.score(test_feature_list,test_class_list)
    else:
        test_accuracy = []

    return test_accuracy

if __name__ == '__main__':
    starttime = time.time()
    print("start")
    folder_path = 'Database/SogouC/Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = text_processing(folder_path,test_size=0.2)

    stopwords_file = 'stopwords_cn.txt'
    stopwords_set = make_word_set(stopwords_file)

    flag = 'sklearn'
    deleteNs = range(0,2000,10)
    test_accuracy_list = []
    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list,deleteN,stopwords_set)
        train_feature_list,test_feature_list = text_features(train_data_list,test_data_list,feature_words,flag)
        test_accuracy = text_classifier(train_feature_list,test_feature_list,train_class_list,test_class_list,flag)
        test_accuracy_list.append(test_accuracy)

    endtime = time.time()
    print(test_accuracy_list)
    print(len(test_accuracy_list))
    plt.plot(deleteNs,test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy alpha=0.38')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()
    print("the cost time:",endtime-starttime)
    print('finished')