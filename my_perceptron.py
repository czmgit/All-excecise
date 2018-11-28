from functools import reduce

class perceptron(object):
    def __init__(self,activatorfunct,train_num,bias):
        self.activator_funct = activatorfunct
        self.train_num = train_num
        self.bias = bias
        self.weights = [0.0 for _ in range(train_num)]
        self.efficiency = 0.2

    def __str__(self):
        return"bias : %f\n,weights : %s\n"%(self.bias,self.weights)

    def predict(self,dataset):
        temp = list(map(lambda x,w:x*w,dataset,self.weights))
        return self.activator_funct(reduce(lambda a,b:a+b,temp,0.0) + self.bias)

    def updata_weights(self,delta,input_vec,rate):
        self.weights = list(map(lambda x,w:rate*delta*x+w,input_vec,self.weights))
        print("weights = ",self.weights)
        self.bias += delta*rate
        print("bias = ",self.bias)

    def train(self,datasets,labels,iteration,rate):
        for i in range(iteration):
            samples = zip(datasets,labels)
            for (input_vec,label) in samples:
                output = self.predict(input_vec)
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
    perceptr = perceptron(afunct, 3, 0.0)
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