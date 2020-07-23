#********************************* 4 min -    min *************************
import torch

x =torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
x_size = x.size()
#x = torch.transpose(x,1)
#x =x.t()
y = torch.tensor([[0.0], [1.0], [4.0], [9.0], [16.0], [25.0], [36.0], [49.0], [64.0], [81.0], [100.0]])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(1,2)
        self.linear2 = torch.nn.Linear(2,1)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        y1 = self.linear1(x)
        y2 = self.relu(y1)
        y3 = self.linear2(y2)
        return y3

model = Model()

loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

print("start")
for i in range(30):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    print(i,loss.item())
    optimizer.zero_grad()
    optimizer.step()

        
    