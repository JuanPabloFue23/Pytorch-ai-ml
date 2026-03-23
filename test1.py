import torch
print(torch.__version__)

import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
model = nn.Sequential(nn.Linear(1, 1))

# Define the optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Prepare data
xs = torch.tensor([[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
ys = torch.tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]], dtype=torch.float32)

# Training loop
for _ in range(500):
    optimizer.zero_grad()
    outputs = model(xs)
    loss = criterion(outputs, ys)
    loss.backward()
    optimizer.step()
    print("Loss = " + str(loss.item()))

# Predict
with torch.no_grad():
    prediction = model(torch.tensor([[10.0]], dtype=torch.float32))
    print(prediction)

# Access the first (and only) layer in the sequential model
layer = model[0]

# Get weights and bias
weights = layer.weight.data.numpy()
bias = layer.bias.data.numpy()

print("\n Final model parameters:")
print("Weights:", weights)
print("Bias:", bias)