from my_perceptron import perceptron

afunct = lambda intx:intx

class lineruint(perceptron):
    def __init__(self,activatorfunct,train_num,bias):
        super.__init__(activatorfunct,train_num,bias)