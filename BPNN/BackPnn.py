from functools import reduce
from math import exp
import random

def sigmoid(x):
    return 1/(1+exp(-x))

class Node(object):
    def __init__(self,layer_index,node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream =[]
        self.upstream =[]
        self.output = 0
        self.delta = 0
    def set_output(self,output):
        self.output = output
    def append_downstream_connection(self,conn):
        self.downstream.append(conn)#添加一个到下游节点的链接
    def append_upstream_connection(self,conn):
        self.upstream.append(conn) #添加一个到上游节点的链接
    def calc_output(self):
        # 语法：reduce(function, iterable[, initializer]),function必须有两个参数，iterable可迭代对象
        #计算 y = sigmoid(w*x+b)
        output = reduce(lambda ret,conn:ret+conn.upstream_node.output*conn.weight,self.upstream,0.0)
        self.output = sigmoid(output)
    def calc_hidden_layer_delta(self):
        #计算隐层误差项 δi= a*（1-a）* ∑(w*δk)  其中a是隐层输出，δk是下一层节点的误差项
        downstream_delta = reduce(lambda ret,conn:ret+conn.downstream_node.delta*conn.weight,self.downstream,0.0)
        self.delta = self.output*(1 - self.output)*downstream_delta
    def calc_output_layer_delta(self,label):
        #计算输出层误差项 δi = y*(1-y)*(t-y)  t是标签，y是实际输出
        self.delta = self.output*(1 - self.output)*(label - self.output)
    def __str__(self):
        node_str = "%u-%u: output:%f delta:%f" % (self.layer_index,self.node_index,self.output,self.delta)
        downstream_str = reduce(lambda ret,conn:ret+'\n\t'+str(conn),self.downstream,'')
        upstream_str = reduce(lambda ret,conn:ret+'\n\t'+str(conn),self.downstream,'')
        return node_str+'\n\tdownstream:'+downstream_str+'\n\tupstream:'+upstream_str

class ConstNode(object):#ConstNode对象，为了实现一个输出恒为1的节点(计算偏置项时需要)
    def __init__(self,layer_index,node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self,conn):
        self.downstream = conn#添加一个到下游节点的连接

    def calc_hidden_layer_delta(self):#根据式4来计算隐层节点的delta
        downstream_delta = reduce(lambda ret,conn:ret+conn.downstream_node.delta * conn.weight,\
                                  self.downstream,0.0)

    def __str__(self):
        node_str = '%u-%u:output:1'%(self.layer_index,self.node_index)
        downstream_str = reduce(lambda ret,conn:ret+'\n\t'+str(conn),self.downstream,'')
        return node_str+'\n\tdownstream:'+downstream_str

class Layer(object):
    def __init__(self,layer_index,node_count):
        #初始化一层,layer_index是层编号，node_count是层包含的节点个数
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index,i))
        self.nodes.append(ConstNode(layer_index,node_count))

    def set_output(self,data):
        #设置层的输出，当层是输入层时会用到
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print(node)

class Connection(object):#记录连接的权重，以及与此连接所关联的上下游节点
    def __init__(self,upstream_node,downstream_node):
        #初始化连接
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1,0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        #计算梯度
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def get_gradient(self):
        #返回梯度
        return self.gradient

    def updata_weight(self,rate):
        #根据梯度下降法更新权重
        self.calc_gradient()
        self.weight += rate*self.gradient

    def __str__(self):
        '''
        打印连接信息
        '''
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)

class Connections(object):#提供connection集合操作
    def __init__(self):
        self.connection = []

    def add_connection(self,connection):
        self.connection.append(connection)

    def dump(self):
        for conn in self.connection:
            print(conn)

class Network(object):#提供API
    def __init__(self,layers):
        #初始化全连接神经网络
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            #创建网络节点
            self.layers.append(Layer(i,layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node,downstream_node)\
                           for upstream_node in self.layers[layer].nodes\
                           for downstream_node in self.layers[layer+1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)#???
                conn.upstream_node.append_downstream_connection(conn)
    def train(self,labels,data_set,rate,epoch):
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],data_set[d],rate)

    def train_one_sample(self,label,sample,rate):
        #内部函数，用一个样本训练网络
        self.predict(sample)
        self.calc_delta(label)
        self.updata_weight(rate)

    def calc_delta(self,label):
        #计算每个节点的delta
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def updata_weight(self,rate):
        #更新每个连接的权重
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.updata_weight(rate)

    def calc_gradient(self):
        #计算每个连接的梯度
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self,label,sample):
        #获得网络在一个样本下的每个连接上的梯度
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self,sample):
        self.layers[0].set_output(sample)
        for i in range(1,len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node:node.output,self.layers[-1].nodes[:-1])

    def dump(self):
        for layer in self.layers:
            layer.dump()

class Normalizer(object):
    def __init__(self):
        self.mask = [0x1,0x2,0x4,0x8,0x10,0x20,0x40,0x80]

    def norm(self,number):
        return list(map(lambda m:0.9if number & m else 0.1,self.mask))

    def denorm(self,vec):
        binary = list(map(lambda i:1 if i>0.5 else 0.5,vec))
        for i in range(len(self.mask)):
            binary[i] = binary[i]*self.mask[i]
        return reduce(lambda x,y:x+y,binary)

def mean_square_error(vec1,vec2):
    return 0.5*reduce(lambda a,b:a+b,map(lambda v:(v[0]-v[1])*(v[0]-v[1]),zip(vec1,vec2)))

def gradient_check(network,sample_feature,sample_label):
    network_error = lambda vec1,vec2:0.5*reduce(lambda a,b:a+b,\
                map(lambda v:(v[0]-v[1])*(v[0]-v[1]),zip(vec1,vec2)))
    network.get_gradient(sample_label,sample_feature)
    for conn in network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()
        epsilon = 0.0001
        # 增加一个很小的值，计算网络的误差
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature),sample_label)

        #减去一个很小的值，计算网络的误差
        # 前面加过一次了，所以减去两倍
        conn.weight -= 2*epsilon
        error2 = network_error(network.predict(sample_feature),sample_label)

        #根据式6计算期望的梯度值
        expected_gradient = (error2 - error1)/(2*epsilon)

        print("expected_gradient:\t%f\nactual gradient:\t%f"%(expected_gradient,actual_gradient))

def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0,256,8):
        n = normalizer.norm(int(random.uniform(0,256)))
        data_set.append(n)
        labels.append(n)
    return labels,data_set

def train(network):
    labels,data_set = train_data_set()
    network.train(labels,data_set,0.3,50)

def test(network,data):
    normalizer = Normalizer()
    norm_data = normalizer.norm(data)
    predict_data = network.predict(norm_data)
    print('\ttestdata(%u\tpredict(%u'%(data,normalizer.denorm(predict_data)))

def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) ==i:
            correct += 1.0
    print('correct_ratio:%.2f%%'%(correct/256*100))

def gradient_check_test():
    net = Network([2,3,4])
    sample_feature = [0.9,0.1]
    sample_label = [0.9,0.1,0.8,0.7]
    gradient_check(net,sample_feature,sample_label)

if __name__ == '__main__':
    gradient_check_test()
    net = Network([8,3,8])
    train(net)
    net.dump()
    correct_ratio(net)