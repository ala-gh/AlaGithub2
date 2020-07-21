#inam baz copy barnamehaye shoma bazam kar nemikone
import pandas as pd

datatrain = pd.read_csv('E:\proposal&payan nameh\projectpython\ProjectHowsamDL\dataskin.csv')
datatrain = datatrain.apply(pd.to_numeric)
datatrain_array = datatrain.values

import numpy as np
idx_test = np.random.randint(0,245056, 5000)
idx_train = list()

for i in range(245056):
    if not i in idx_test:
        idx_train.append(i)
datatest_array = datatrain_array[idx_test,:]

#split x and y (feature and target)
xtest = datatest_array[:,:3]
ytest = datatest_array[:,3]

datatrain_array = datatrain_array[idx_train,:]
#split x and y (feature and target)
xtrain = datatrain_array[:,:3]
ytrain = datatrain_array[:,3]

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1234)


hl = 10
lr = 0.01
num_epoch = 500


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, hl)
        self.fc2 = nn.Linear(hl, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
X = torch.Tensor(xtrain).float()
Y = torch.Tensor(ytrain).long()

Xv = torch.Tensor(xtest).float()
Yv = torch.Tensor(ytest).long()
#train
for epoch in range(num_epoch):

    optimizer.zero_grad()
    print("optimizer zero")
    out = net(X)
    loss = criterion(out, Y)
    print("epoch:",epoch,loss.item())
    loss.backward()
    optimizer.step()


