#********************************* 4 min -    min *************************
import torch

x1 =torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
x_size = x1.size()
y1 = torch.tensor([[0.0], [1.0], [4.0], [9.0], [16.0], [25.0], [36.0], [49.0], [64.0], [81.0], [100.0]])
#y1 = torch.tensor([[0.0], [1.0], [4.0], [6.0], [8.0], [10.0], [12.0], [14.0], [16.0], [18.0], [20.0]])

class Model(torch.nn.Module):
    def __init__(self,hidden_layer):
        super(Model, self).__init__()
        self.linear0 = torch.nn.Linear(1,1)
        self.linear1 = torch.nn.Linear(1,hidden_layer)
        self.linear2 = torch.nn.Linear(hidden_layer,1)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        y0 = self.linear0(x)
        y1 = self.linear1(x)
        y2 = self.relu(y1)
        y3 = self.linear2(y2)
        #return y3
        return y0
        #return  y1
hl =2
model = Model(hl)
learning_rate = 0.01
loss_fn = torch.nn.MSELoss(size_average = False)  #size_average = False    #reduction='sum'
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

print("start")
model.train()
epoch =100
for i in range(epoch):

    y_pred = model(x1)
    loss = loss_fn(y_pred,y1)
    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if loss < 1e-3:
        print("finish")
        break

print("epoch :",epoch,"\nlearning rate :", learning_rate,"\nhidden layer : ",hl)
#model.eval()

# x2 =torch.tensor([[12.0], [14.0], [2.0], [16.0], [4.0], [5.0], [20.0], [7.0], [25.0], [9.0], [10.0]])
#
# y2 = model(x2)
# print(y2)
#
    