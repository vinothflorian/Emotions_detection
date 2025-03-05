from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
from PIL import Image


def metric(truevalue, predictedvalue):
    print(f"Accuracy : {accuracy_score(truevalue, predictedvalue)}")
    print(f"Precision : {precision_score(truevalue, predictedvalue, average= 'macro')}")
    print(f"Recall : {recall_score(truevalue, predictedvalue, average= 'macro')}")
    print(f"f1_Score: {f1_score(truevalue, predictedvalue, average= 'macro' )}")

transform  = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=r"D:\PYTHON\VSCODE-PY\emotions\Emotions_1\train",transform= transform)
#test_dataset = datasets.ImageFolder(root=r"D:\Python Projects\Production\Final Project\Emotions Detection\Dataset\dataset\test",transform= transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=0)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=0)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=0)
        self.conv5 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.dropout=nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*1*1,256)
        self.fc2 = nn.Linear(256,7)

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = torch.relu(self.pool(X))
        X = self.conv3(X)
        X = torch.relu(self.pool(X))
        X = self.conv4(X)
        X = torch.relu(self.pool(X))
        X = self.conv5(X)
        X = torch.relu(self.pool(X))
        X = torch.flatten(X,1)
        X = self.fc1(X)
        X= self.dropout(X)
        X = self.fc2(X)
        return X
    

model = EmotionClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
  

n_epoch= 20
for epoch in range(1,n_epoch+1):
  running_loss = 0.0
  correct=0
  total=0
  train_true = []
  train_pred = []
  for inputs,labels in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    running_loss+=loss.item()
    total=total+labels.size(0)
    _,predictions=torch.max(outputs,1)
    correct=correct+ (predictions==labels).sum().item()
    train_true.extend(labels)
    train_pred.extend(predictions)
  print(f"Epoch: {epoch}/{n_epoch} Loss: {running_loss/len(train_loader)}")
  print(f"Epoch: {epoch}/{n_epoch} Accuracy: {correct/total}")
print('*******Train**********')
metric(train_true, train_pred)

torch.save(model.state_dict(), r"D:\PYTHON\VSCODE-PY\emotions\emotion.pth")
print("Model saved to emot_train.pth")