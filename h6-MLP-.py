#hata copy paste barname gozashtam bazam hal nashod.
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split, DataLoader
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(3,5)
        self.linear2 = nn.Linear(5,1)

    def forward(self, x):
        y1 = self.linear1(x)
        y2 = F.relu(y1)
        y3 = self.linear2(y2)
        return y3

class Data(Dataset):
    def __init__(self,dataset_path):
        self.data = pd.read_csv(dataset_path)
        self.data_numeric = self.data.apply(pd.to_numeric)
        self.data_array = self.data_numeric.values
        #print(self.data_array[1])
        self.x = torch.Tensor(self.data_array[:, :3]).float()
        self.y = torch.Tensor(self.data_array[:, 3]).float()

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

dataloader = DataLoader(data, batch_size=10, shuffle=True)
print("created dataloader")
loss_function = torch.nn.MSELoss(reduction='sum')
#loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

# for data_batch, label_batch in dataloader:
#     x = data_batch
#     y = label_batch

for epoch in range(2) :
    for i , dataa in enumerate(dataloader):
        inputs , labels = dataa
        inputs = Variable(inputs)
        labels = Variable(labels)
        y_pred = model(inputs)
        y_pred = Variable(y_pred, requires_grad=True)
        #torch.reshape(labels,(10,1))
        #print(len(y_pred),len(labels))
        #print(type(y_pred),type(labels))
        #print(y_pred.shape,labels.shape)
        loss = loss_function(y_pred,labels)
        print(epoch,loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




# for i ,data_batch, label_batch in dataloader :
#         optimizer.zero_grad()
#         print("shape",data_batch.shape)
#         out = model(data_batch)
#         loss = optimizer(out,label_batch)
#         print(i,loss.item())
#         loss.backward()
#         optimizer.step()







