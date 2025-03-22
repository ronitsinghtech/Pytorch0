import torch
import torch.nn as nn
import numpy as np

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1,1)    # 1 Input, 1 output

    def forward(self, x):
        return self.linear(x)
x_train = np.random.rand(100, 1).astype(np.float32) # 100 random values between 0 and 1

y_train = 2 * x_train + 1 + np.random.randn(100, 1).astype(np.float32) * 0.1   # y = 2x + 1 + noise

#converting to pytorch tensors.

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# Setting loss 
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.01) # lr = learning rate = how large of a step you want to do in the model. Smaller step means more accuracy, but takes much longer.

for epoch in range(10000):
    optimizer.zero_grad()   #Clear the gradients
    outputs = model(x_train)    #Forward Pass (Figuring out outputs of the current variation of the model)
    loss = criterion(outputs, y_train)  #Compute Loss
    loss.backward()     # Backward pass
    optimizer.step()    # Update weights (Through one step forward)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

x_test = torch.tensor([[0.5], [1.0], [1.5]])
predictions = model(x_test)
print("Predictions: ", predictions.detach().numpy())

