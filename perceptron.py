from functools import reduce

class perceptron(object):
    def __init__(self,input_num,activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        return'weights1111\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self,input_vec):#算法：y = f(x*w+b)
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
        #temp = reduce(lambda a,b:a+b,list(map(lambda x,w:x*w,list(zip(input_vec,self.weights)))),0.0) + self.bias
        temp =  list(map(lambda x,w:w*x,input_vec,self.weights))
        return self.activator(reduce(lambda a,b:a+b,temp,0.0) + self.bias)

    def train(self,input_vecs,labels,iteration,rate):#输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self,input_vecs,labels,rate):
        #一次迭代，把所有的训练数据过一遍
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = zip(input_vecs,labels)
        for (input_vec,label) in samples:
            #计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            #更新权重
            self._updata_weights(input_vec,output,label,rate )

    def _updata_weights(self,input_vec,output,label,rate):
        #按照感知器的规则来更新权重
        delta = label - output
        self.weights = list(map(lambda x,w:w+rate*delta*x,input_vec,self.weights))
        print("weights = ",list(self.weights))
        self.bias += rate*delta


def f(x):
    return 1 if x > 0 else 0

def get_training_dataset():
    input_vec = [[1,1],[0,0],[1,0]]
    labels = [1,1,0]
    return input_vec,labels

def train_and_perceptron():
    # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    p = perceptron(2,f)
    # 训练，迭代10轮, 学习速率为0.1
    input_vec,labels = get_training_dataset()
    p.train(input_vec,labels,10,0.1)
    return p

if __name__ == '__main__':
    and_perceptron = train_and_perceptron()
    print(and_perceptron)
    #测试
    print('1 and 1 = %d ' % and_perceptron.predict([1, 1]))
    print('1 and 0 = %d ' % and_perceptron.predict([1, 0]))
    print('0 and 1 = %d ' % and_perceptron.predict([5, 3]))
    print('0 and 0 = %d ' % and_perceptron.predict([0, 0]))
