#*******************************import library**********************************
import pandas as pd
import numpy as np
import torch
from random import randint
from scipy.spatial import distance


#********************************calculate distance function*******************
'''
Calculate the distance of each data from each center:
It has two loops. The first loop considers all data. The second loop for each data from Previous loop calculate distance from each center.
To whichever is closer, the index  of center saved in minIndex.
finally all index saved in labels list and return it. 
'''
def minDistance(datasett, centers):
    labels = []
    for i in range(len(datasett)):
        distancee = []
        for j in range(len(centers)):
            d1 = distance.euclidean(datasett[i],centers[j])
            distancee.append(d1)
        minIndex = np.argmin(distancee)
        labels.append(minIndex)
    return labels

#***************************update centers function*****************************
'''
update centers:
we have k centers.also We have as many labels as the data that each value of the label 
represents the central index with the shortest distance to the data.
for example 'labels[4] = 0 ' means The closest center(from k centers) to the Fifth data is the first center
It has two loops. The first loop considers all centers.The second loop for each center data from Previous loop 
averaged all the data whose labels belong to that center.And that average replaces the center and finally return it.
'''
def updateCenters(k,labels,datasett):
    c1 = []
    for i in range(k):
        calculate = []
        for j in range(len(labels)):
            if labels[j] == i:
                calculate.append(datasett[j])
        calculateTensor = torch.stack(calculate)
        cNew = torch.mean(calculateTensor, 0)
        c1.append(cNew)

    return c1


#*********************************get the dataset****************************

'''
Preprocessing data and create primary centers:
read data from address, Convert it to number and tensor, Delete the last column mean labels,
 select k  random data as centers.
 finally return data and centers
'''
def getData(adress,k):

    data = pd.read_csv(adress)
    data_array = data.values
    data_array = data_array[:,:-1]
    dataset = np.array(data_array,dtype=np.float32)
    datasetTensor =  torch.from_numpy(dataset)
    x,y = datasetTensor.shape
    c = []
    for i in range(k):
        rand = randint(0, x)
        c.append(datasetTensor[rand])
    cTensorRandom = torch.stack(c)
    return datasetTensor, cTensorRandom


#************************************ main ************************************
adress = "E:\proposal&payan nameh\projectpython\ProjectHowsamDL\iris.csv"
k =3
datasetTensor, cTensor = getData(adress,k)
for i in range (10):
    print("***********************************")
    print('num =', i)

    label = minDistance(datasetTensor, cTensor)
    cnew = updateCenters(k, label, datasetTensor)
    print(label)
    print(cnew)
    cTensor = cnew
#****************************FINISH*******************************************












# #**************************mean a list function**************************
# def mean(calculate):
#     calculateTensor = torch.stack(calculate)
#     x, y = calculateTensor.shape
#     for m in range(len(calculateTensor)):
#         for j in range(y):
#             sum = 0
#             for i in range(x):
#                 sum = sum + calculateTensor[i,j]
#             cTensor[m] = sum / len(calculateTensor)
#     return cTensor

#cNew=mean(calculateTensor)


#


