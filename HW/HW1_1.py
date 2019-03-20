import pandas,math,os,csv
from functools import reduce
import numpy as np

freamfile = 'model.npy'

def getdataset():
    trainlist = []
    traindatas = [[]for _ in range(18)]
    csvdata = pandas.read_csv('train.csv',  encoding='big5')
    train = csvdata.values.tolist()
    for i in range(len(train)):
        trainlist.append(train[i][3:])

    lenght1 = len(trainlist[0])
    lenght2 = len(trainlist)
    for i_row in range(lenght2):
        if (i_row % 18) == 10:
            for i in range(lenght1):
                if trainlist[i_row][i] == "NR":
                    trainlist[i_row][i] = 0
        temp1 = list(map(float,trainlist[i_row]))
        traindatas[i_row%18] += temp1
    output_y = []
    input_x = []
    for k in range(12):
        for i in range(471):
            input_x.append([])
            for j in range(18):
                for t in range(9):
                    input_x[k*471+i].append(traindatas[j][k*471+i+t])
            output_y.append(traindatas[9][k*480+i+9])
    output_y = np.array(output_y)
    input_x = np.array(input_x)
    temp = np.random.randint(1,2,size = [len(input_x),1])
    # 将偏置加入输入中，concatenate将两个数组拼接起来，axis=1是列拼接，axis是行拼接
    input_x = np.concatenate((temp,input_x),axis =1)
    return output_y,input_x

def datatrain(y,x):
    lenx = len(train_x[0])
    w = np.zeros(lenx)
    x_t = train_x.transpose()#将输入转置
    s_gra = np.zeros(lenx)
    l_rate = 0.001
    iteration = 1000
    """
    for i_tera in range(iteration):
        hypo = np.dot(x, w)
        loss = hypo - y
        cost = np.sum(loss ** 2) / len(x)
        cost_a = math.sqrt(cost)
        gra = np.dot(x_t, loss)
        s_gra += gra ** 2
        ada = np.sqrt(s_gra)
        w = w - l_rate * gra / ada
        print('iteration: %d | Cost: %f  ' % (i_tera, cost_a))
    """
    for i_tera in range(iteration):
        for i in range(len(x)):
            hypo = np.dot(x[i], w)
            error = y[i] - hypo
            xt = x[i].transpose()  # 将输入转置
            gra = error*xt#np.dot(train_xt, error)  # 计算梯度
            s_gra += gra ** 2
            ada = np.sqrt(s_gra)
            w -= l_rate * gra / ada  # 更新权重
            loss = error ** 2#np.sum(error ** 2) / len(error)  # 计算损失函数
            loss_a = math.sqrt(loss)  # ????
            if loss_a < 0.11:
                return loss_a
            print("Iteration:%d | Loss:%f" % (i_tera, loss_a))

    np.save(freamfile,w)

def read_testdata():
    test_x =[]
    test_data = []
    testlist = [[] for _ in range(18)]
    csvdata = pandas.read_csv('test.csv', header=None, encoding='big5')
    train = csvdata.values.tolist()
    for i in range(len(train)):
        test_data.append(train[i][2:])
        if i%18 == 10:
            for j in range(len(test_data[0])):
                if test_data[i][j] =='NR':
                    test_data[i][j] = 0
        temp = list(map(float, test_data[i]))
        testlist[i%18] += temp
    for i in range(240):
        test_x.append([])
        for j in range(18):
            for k in range(9):
                test_x[i].append(testlist[j][i*9+k])
    temp = np.random.randint(1,2,size=[len(test_x),1])
    test_x = np.concatenate((temp,test_x),axis=1)
    return test_x

def TestAndSave(testdata):
    weight = np.load(freamfile)
    ans = []
    for i in range(len(testdata)):
        ans.append(["id_"+ str(i)])
        output = np.dot(testdata[i],weight)
        ans[i].append(output)
    filename = "predict.csv"
    text = open(filename,"w+")
    s = csv.writer(text,delimiter =",",lineterminator ='\n')
    s.writerow(['id','value'])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()

if __name__ == '__main__':
    train_y,train_x= getdataset()
    datatrain(train_y,train_x)
    testdata = read_testdata()
    TestAndSave(testdata)