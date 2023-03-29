import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

transform  = transforms.Compose([
                                 transforms.Lambda(lambda image: image.convert('RGB')),
                                 transforms.Resize((64,64)),
                                 transforms.ToTensor()
])

train_data = datasets.MNIST(root='./Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./Data', train=False, download=False, transform=transform)

print(train_data)
print()
print(test_data)

image,label = train_data[0]
print(image.shape)
print(label)

torch.manual_seed(42)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

model = torchvision.models.alexnet(pretrained=True)
print(model)

for param in model.parameters():
  param.requires_grad = False

model.classifier[6] = nn.Linear(4096, 10)
model.classifier.add_module('7', nn.LogSoftmax(dim=1))
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = 'cpu'
model.to(device)

import time

start_time = time.time()

epochs = 2
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
  train_losses[i] = train_losses[i].cpu().detach().numpy()

for i in range(len(test_losses)):
  test_losses[i] = test_losses[i].cpu().detach().numpy()

print("train loss: ", train_losses)
print("test_loss : ", train_losses)

plt.plot(train_losses, label='train_loss')
plt.plot(test_losses, label='validation_loss')
plt.ylabel('Losses')
plt.xlabel('epoch')
plt.xticks(range(0,5))
plt.legend()
print(plt.show())

torch.manual_seed(42)
test_loader_all = DataLoader(test_data, batch_size=10000, shuffle=False)
test_loader_mini = DataLoader(test_data, batch_size=100, shuffle=False)

preds = []

with torch.no_grad():
  correct = 0
  for batch, (X_test, y_test) in enumerate(test_loader_mini):
    batch+=1
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    y_val = model(X_test)
    predicted = torch.max(y_val, 1)[1]
    preds.append(predicted)
    
    correct+=(predicted==y_test).sum()

y_test = test_data.targets
y_test = y_test.numpy()

y_pred = []

for batch_y in preds:
  for y_val in batch_y:
    y_pred.append(y_val)

for i in range(len(y_pred)):
  y_pred[i] = y_pred[i].cpu().detach().numpy()

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))