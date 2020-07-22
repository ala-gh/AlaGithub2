#hata copy paste barname gozashtam bazam hal nashod.
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split, DataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(3,5)
        self.linear2 = nn.Linear(5,2)

    def forward(self,x):
        y1 = self.linear1(x)
        y2 = F.relu(y1)
        y3 = self.linear2(y2)
        return y3

class Data(Dataset):
    def __init__(self,dataset_path):
        self.data = pd.read_csv(dataset_path)
        self.data_numeric = self.data.apply(pd.to_numeric)
        self.data_array = self.data_numeric.values
        print(self.data_array[1])
        self.x = torch.Tensor(self.data_array[:, :3]).float()
        self.y = torch.Tensor(self.data_array[:, 3]).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = (self.x[idx,:], self.y[idx])
        return sample

print("start")
model = Net()
print("created model")
data = Data('E:\proposal&payan nameh\projectpython\ProjectHowsamDL\dataskin.csv')
print("created data")
print("len",len(data))
print("data[4]",data[4])

number_of_train=(0.7*len(data)).__int__()
number_of_test=(0.2*len(data)).__int__()
number_of_valid=(0.1*len(data)).__int__()
print("number_of_valid",number_of_valid)

train, valid, test = random_split(data,[number_of_train, number_of_valid, number_of_test])    #?? inja error dare ... chera ?
print("self.train",train)

dataloader = DataLoader(train, batch_size=10, shuffle=True)
print("created dataloader")
loss_function = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for data_batch, label_batch in dataloader:
    x = data_batch
    y = label_batch
for i ,data_batch, label_batch in data :
        optimizer.zero_grad()
        print("shape",data_batch.shape)
        out = model(data_batch)
        loss = optimizer(out,label_batch)
        print(i,loss.item())
        loss.backward()
        optimizer.step()







