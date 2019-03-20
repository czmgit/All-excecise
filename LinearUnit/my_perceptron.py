from functools import reduce

class perceptron(object):
    def __init__(self,activatorfunct,train_num):
        self.activator_funct = activatorfunct
        self.train_num = train_num
        self.bias = 0.0
        self.weights = [0.0 for _ in range(train_num)]
        self.efficiency = 0.2

    def __str__(self):
        return"bias : %f\n,weights : %s\n"%(self.bias,self.weights)

    def output(self,input_vec):
        #map(function, iterable) ：对iterable中的item依次执行function(item)，返回结果是一个list地址，需转成list
        temp = tuple(map(lambda x,w:x*w,input_vec,self.weights))
        #语法：reduce(function, iterable[, initializer]),function必须有两个参数，iterable可迭代对象，如列表，元组
        return self.activator_funct(reduce(lambda a,b:a+b,temp,0.0) + self.bias)

    def updata_weights(self,delta,input_vec,rate):
        self.weights = list(map(lambda x,w:rate*delta*x+w,input_vec,self.weights))
        print("weights = ",self.weights)
        self.bias += delta*rate
        print("bias = ",self.bias)

    def train(self,datasets,labels,iteration,rate):
        for i in range(iteration):
            samples = zip(datasets,labels)
            for input_vec,label in samples:
                output = self.output(input_vec)
                delta = label - output
                self.updata_weights(delta,input_vec,rate)

def afunct(intx):
    return 1 if intx>0 else 0

def get_dataset():
    datasets = [[1,1,1],[1,1,0],[1,0,1],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[0,0,0]]
    labels = [1,0,1,0,1,0,1,1]
    return datasets,labels

def data_train():
    datasets,labels = get_dataset()
    perceptr = perceptron(afunct, 3)
    perceptr.train(datasets,labels,10,0.1)

    return perceptr

def get_testdataset():
    test_dataset = [
        [1, 1, 2],[1, 0, 2],[5, 3, 0],[0, 2, 1],
        [0, 1, 2],[0, 3, 0],[0, 0, 1],[0, 0, 0]
    ]
    return test_dataset

def data_test(P):
    datasets = get_testdataset()
    for i in range(len(datasets)):
        result = P.predict(datasets[i])
        print("%s:\t%d\n" % (datasets[i],result))

if __name__ == '__main__':
    percept = data_train()#1 098 927 832
    print(percept)
    data_test(percept)