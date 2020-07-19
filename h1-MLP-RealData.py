import pandas as pd
import torch

Model = torch.nn.Sequential(
    torch.nn.Linear(3,5),
    torch.nn.ReLU(),
    torch.nn.Linear(5,2)
)

data = pd.read_csv("E:\proposal&payan nameh\projectpython\p19-howsam-MLP-realdata\dataskin.csv")
data_numeric = data.apply(pd.to_numeric)
data_array = data_numeric.values
#data_array_len = len(data_array)
import numpy as np

import numpy as np
idx_test = np.random.randint(0, 245056, 5000)
idx_train = list()
a =245056-5000
print("a",a)
b =0
for i in range(245055):
    if not i in idx_test:
        idx_train.append(i)
        b =b+1
print("b",b)
print("idx:",idx_test,"len:",len(idx_test))
print("len",len(idx_train))


loss_function = torch.nn.MSELoss(reduction='sum')
new_w = torch.optim.SGD(Model.parameters, lr=0.01)

from torch.utils.data import random_split, DataLoader
data_train, data_valid, data_test = random_split(data_array, (171540, 49011, 24505))
#x_train = data_train[:,:3]
#y_train = data_train[:,3]

dataloader = DataLoader(data_train, batch_size=10, shuffle=True)

print(type(dataloader))

for data_batch, label_batch in dataloader:
    new_w.zero_grad()

    out = Model(data_batch)
    loss = loss_function(out, label_batch)
    print("loss:",loss.item())
    loss.backward()
    new_w.step()




#----------------------------------1--------------------------------------------
# data_array_len = len(data_array)
# print(data_array_len)
# print("**",data_array_len*0.2)
# print("***",round(data_array_len*0.2))
# import numpy as np
# idx_valid = np.random.randn(0,data_array_len,round(data_array_len*0.2))
# print(len(idx_valid))
#
# idx_train = list()
# idx_test = list()
#
# for i in range(round(data_array_len*0.7)):
#     if i not in idx_valid:
#         idx_train.append(i)
#
