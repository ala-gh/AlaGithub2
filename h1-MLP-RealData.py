import pandas as pd
import torch
import torch.nn as nn
torch.manual_seed(1234)   #? howsam - yani chi ?
import torch.nn.functional as F

Model = torch.nn.Sequential(
    torch.nn.Linear(3, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 2)
)

data = pd.read_csv("E:\proposal&payan nameh\projectpython\ProjectHowsamDL\dataskin.csv")
data_numeric = data.apply(pd.to_numeric)
data_array = data_numeric.values

#data_array_len = len(data_array)

import numpy as np
idx_test = np.random.randint(0, 245056, 5000)
idx_train = list()

for i in range(245056):
    if not i in idx_test:
        idx_train.append(i)

data_train = data_array[idx_train,:]
data_test = data_array[idx_test,:]

x_train = data_train[:,:3]
y_train = data_train[:,3]

x_test = data_test[:,:3]
y_test = data_test[:,3]

#without shufle ....

#loss_function = torch.nn.MSELoss(reduction='sum')
loss_function = nn.CrossEntropyLoss()
new_w = torch.optim.SGD(Model.parameters(), lr=0.01)

for i in range(100):
    out = Model(x_train)    # howsam - ?? inja error dare...
    loss = loss_function(out, y_train)
    print("loss", loss.item())
    new_w.zero_grad()
    loss.backward()
    new_w.step()

print("ala")
#-------------------------------------------------------------------

# from torch.utils.data import random_split, DataLoader
# data_train, data_valid, data_test = random_split(data_array, (171540, 49011, 24505))
# #x_train = data_train[:,:3]
# #y_train = data_train[:,3]
#dataloader = DataLoader(data_train, batch_size=10, shuffle=True)

# for data_batch, label_batch in dataloader:
#     new_w.zero_grad()
#
#     out = Model(data_batch)
#     loss = loss_function(out, label_batch)
#     print("loss:",loss.item())
#     loss.backward()
#     new_w.step()

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
