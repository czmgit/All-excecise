import numpy as np
from matplotlib import pyplot as plt
def randomInitialize():
    N = 5
    M = 4
    K = 3
    ndz = np.random.randint(0,10,size=[N, K])+2
    nzw = np.random.randint(0,10,size=[K, M])+0.7
    nz = np.random.randint(0,10,size=[K])+0.3
    topic_num = 50
    list = [1,2,3,4,5,6,7,8,9];
    print(list[-4:])
    perplexity = {'1':100,'2':300,'3':600,'4':800,'5':1200,'6':1600}
    plt.plot(perplexity.keys(), perplexity.values())
    plt.title("Topic Number %d" % topic_num)
    plt.xlabel("Iteration Count")
    plt.ylabel("Perplexity")
    plt.show()
    for d in range(5):
        for w in range(4):
            print("ndz",ndz[d, :])
            print("nzw",nzw[:, w])
            print("nz",nz)
            print("multiply",np.multiply(ndz[d,:],nzw[:,w]))
            print("divide",np.divide(np.multiply(ndz[d, :], nzw[:, w]), nz))
            pz = np.divide(np.multiply(ndz[d, :], nzw[:, w]), nz)
            print("sum",pz.sum())
            print(pz/pz.sum())
            print("random",np.random.multinomial(1, pz / pz.sum()))
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            print("z=",z)
            print("\n")
            ndz[d, z] += 1
            nzw[z, w] += 1
            nz[z] += 1

if __name__ == '__main__':
    randomInitialize()