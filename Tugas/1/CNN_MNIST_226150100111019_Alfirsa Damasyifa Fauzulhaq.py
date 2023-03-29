import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

transform  = transforms.ToTensor()

train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../Data', train=False, download=False, transform=transform)

print(train_data)
print()
print(test_data)

image,label = train_data[0]
print(image.shape)
print(label)

print(plt.imshow(image.reshape((28,28)), cmap='gray'))
print(plt.show())

torch.manual_seed(42)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

class ConvMNIST(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,16,3,1)
    self.conv2 = nn.Conv2d(16,32,3,1)
    self.conv2_bn = nn.BatchNorm2d(32)
    self.fc1 = nn.Linear(5*5*32, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)

    self.dropout = nn.Dropout(.4)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2, 2)
    x = self.conv2(x)
    x = self.conv2_bn(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2, 2)

    x = x.view(-1, 32*5*5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    x = self.fc3(x)

    return F.log_softmax(x, dim=1)

torch.manual_seed(42)

device = torch.device("cpu")
model = ConvMNIST()
model.to(device)

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()

epochs = 5
train_losses = []
test_losses = []
train_corrects = []
test_corrects = []

for i in range(epochs):

  train_correct = 0
  for batch, (X_train, y_train) in enumerate(train_loader):
    batch+=1

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    predicted = torch.max(y_pred.data, 1)[1]
    batch_correct = (predicted==y_train).sum()
    train_correct+=batch_correct

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch%256 == 0:
      print(f"Epoch {i+1} Batch {batch} loss: {loss.item()}")

  train_losses.append(loss)
  train_corrects.append(train_correct)
  
  test_correct = 0
  with torch.no_grad():
    for batch, (X_test, y_test) in enumerate(train_loader):
      
      X_test = X_test.to(device)
      y_test = y_test.to(device)

      y_val = model(X_test)

      predicted = torch.max(y_val.data,1)[1]

      test_correct+=(predicted==y_test).sum()
  
  loss = criterion(y_val, y_test)
  test_losses.append(loss)
  test_corrects.append(test_correct)

current_time = time.time()
total = current_time - start_time
print(f"Training finished in {total/60} minutes")

for i in range(len(train_losses)):
  train_losses[i] = train_losses[i].detach().numpy()

for i in range(len(test_losses)):
  test_losses[i] = test_losses[i].detach().numpy()

plt.plot(train_losses, label='train_loss')
plt.plot(test_losses, label='validation_loss')
plt.ylabel('Losses')
plt.xlabel('epoch')
plt.xticks(range(0,5))
plt.legend()
print(plt.show())

torch.manual_seed(42)
test_loader_all = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
  correct = 0
  for X_test, y_test in test_loader_all:
    y_val = model(X_test)
    predicted = torch.max(y_val, 1)[1]
    correct+=(predicted==y_test).sum()

print(correct.item()/len(test_data))

print(classification_report(y_test, predicted))

print(confusion_matrix(y_test, predicted))