import torch
import torch.nn as nn
import torch.optim as optim
from XORModel import XORModel

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype = torch.float32)
print(X)

y = torch.tensor([0,1,1,0],dtype = torch.float32)
print(y)

model = XORModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

#training the model

for epoch in range(100):
    #forward pass
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)

    #backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(outputs)

