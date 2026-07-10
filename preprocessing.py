import numpy as np

def new_dataset(dataset, step_size):

    dataX = []
    dataY = []

    for i in range(len(dataset)-step_size-1):

        a = dataset[i:(i+step_size),0]

        dataX.append(a)

        dataY.append(dataset[i+step_size,0])

    return np.array(dataX), np.array(dataY)