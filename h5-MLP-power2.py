#********************************* 4 min -    min *************************
import torch

x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) #.t()
#x = torch.transpose(x,1)
y = torch.tensor([0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(1,1)

    def forward(self,x):
        y1 = self.linear1(x)
        return y1

model = Model()
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for i in range(30):
    y_pred = model(x)    #?? howsam  -inja error dare
    loss = loss_fn(y_pred,y)
    print(i,loss.item())
    optimizer.zero_grad()
    optimizer.step()

        
    