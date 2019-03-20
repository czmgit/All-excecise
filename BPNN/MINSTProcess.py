import struct
import BackPnn as bp
from datetime import datetime

class Loader(object):
    def __init__(self,path,count):
        #初始化加载器，path数据加载路径，count文件中的样本个数
        self.path = path
        self.count = count

    def getfilecontent(self):
        #读取文件内容
        f = open(self.path,'rb')
        content = f.read()
        f.close()
        return content
    def toint(self):
        return struct.unpack('B',byte)[0]#这个函数是什么作用?待验证

class ImageLoader(Loader):
    def getpicture(self,content,index):
        #内部函数，从文件中获取图像
        start = index*28*28+16#为什么要加16？
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(self.toint(content[start+i*28+j]))
        return picture
    def getonesample(self,picture):
        #内部函数，将图像转化为样本的输入向量
        sample =[]
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample
    def load(self):
        #加载数据文件。获得全部样本的输入向量
        content = self.getfilecontent()
        dataset = []
        for index in range(self.count):
            dataset.append(self.getonesample(self.getpicture(content,index)))
        return dataset

class LabelLoader(Loader):
    def load(self):
        #加载数据文件，获得全部样本的标签向量
        content = self.getfilecontent()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index+8]))
        return labels

    def norm(self,label):
        #内部函数，将一个值转换为10维标签向量
        label_vec = []
        label_value = self.toint(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec

def get_training_dataset():
    #获取训练数据集
    image_loader = ImageLoader('train-images-idx3-ubyte',60000)
    label_loader = LabelLoader('train-label-idx1-ubyte',60000)
    return image_loader.load(),label_loader.load()

def get_test_dataset():
    #获取测试数据集
    image_loader = ImageLoader('t10k-images-idx3-ubyte', 10000)
    label_loader = LabelLoader('t10k-label-idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()

def get_result(vec):
    #从10维的向量中取出最大的那个数
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

def evaluate(network,test_dataset,test_labels):
    #错误率计算
    error = 0
    total = len(test_dataset)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_dataset[i]))
        if label != predict:
            error +=1
    return float(error)/float(total)

def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_dataset,train_labels = get_training_dataset()
    test_dataset,test_labels = get_test_dataset()
    network = bp.Network([784,300,10])
    while True:
        epoch += 1
        network.train(train_labels,train_dataset,0.3,1)
        print('%s epoch %d finished' % (datetime(),epoch))
        if epoch%10 == 0:
            error_ratio = evaluate(network,train_dataset,test_labels)
            print('%s after epoch %d,error ratio is %f'%(datetime(),epoch,error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio

if __name__ == '__main__':
    train_and_evaluate()

