from my_perceptron import perceptron
import matplotlib.pyplot as plt

afunct = lambda intx:intx

class lineruint(perceptron):
    def __init__(self,afunct,train_num):
        perceptron.__init__(self,afunct,train_num)

def get_data():
    datasets = [[5],[3],[8],[1.4],[10.1]]
    labels = [5500,2300,7600,1800,11400]
    return datasets,labels

def get_testdata():
    test_data = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[20],[25],[30],[35],[40],[45],[50]]
    return test_data

def train_linear(afunct,train_num):
    datasets,labels = get_data()
    lu = lineruint(afunct,train_num)
    lu.train(datasets,labels,10,0.01)
    return lu

def test_data(linear_unit):
    testdatas = get_testdata()
    test_result = []
    for i in range(len(testdatas)):
        test_result.append(linear_unit.output(testdatas[i]))
        print('Work %d years, monthly salary = %.1f'% (testdatas[i][0],test_result[i]))

    plt.plot(testdatas,test_result,linewidth = 5)
    plt.title("Monethly Salary For Work Year")
    plt.xlabel("Work year")
    plt.ylabel("Monthly salary")
    plt.show()

if __name__ == '__main__':
    linear_unit = train_linear(afunct,5)
    print("linear_unit:",linear_unit)
    test_data(linear_unit)
