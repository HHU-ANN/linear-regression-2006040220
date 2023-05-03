# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x,y=read_data()
    lam=2e-10
    weight=np.matmul(np.linalg.pinv(np.matmul(x.T,x)+np.matmul(,np.eye(6))),np.matmul(x.T,y))
    return weight @ data
    pass
    
def lasso(data):
    x, y = read_data()
    weight = np.ones(6)
    rate = 1e-12
    label = 1e-5
    for i in range(int(1e10)):
        Y = np.matmul(weight, x)
        loss = np.sum(Y - y) ** 2 + np.linalg.norm(weight,ord=1)
        if loss < label
            break
        dweight=np.matmul((Y - y),x.T)
        weight=weight-rate * dweight
    return weight @ data
    pass

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y