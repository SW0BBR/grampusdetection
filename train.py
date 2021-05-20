import torch, torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from datetime import datetime
from torchsummary import summary

# Defining model parameters
batch_size = 16
epochs = 18
num_classes = 9
learning_rate = 0.001

# Defining model
class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 224, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(224, 256, kernel_size=3,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 150 x 16 x 16

            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(512 * 12 * 12, 800),
            nn.ReLU(),
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(600, num_classes))
        
    def forward(self, xb):
        return self.network(xb) 

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



# Load imagefolder
print("Loading dataset ...")
data_dir = "/media/alex/DOLPHINDRIV/data/transforms_test"
# data_dir = "/home/alabro/dd_input/transforms"
classes = sorted(os.listdir(data_dir))

# Convert to format that pytorch can work with
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize
    (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

dolphin_dataset = torchvision.datasets.ImageFolder(data_dir, transform=train_transform)

# Split into testing, validation and training set
print("Splitting dataset ...")
val_size = int(0.15 * len(dolphin_dataset))
test_size = val_size
train_size = len(dolphin_dataset) - 2 * val_size
# dd_train, dd_test = torch.utils.data.random_split(dolphin_dataset, [train_size, test_size])
dd_train, dd_val, dd_test = torch.utils.data.random_split(dolphin_dataset, [train_size, val_size, test_size])
print(dd_train)

# Create dataloaders for quick access
print("Creating dataloaders ...")
dd_trainloader = torch.utils.data.dataloader.DataLoader(dd_train, batch_size, shuffle=True, num_workers=4, pin_memory=True)
dd_testloader = torch.utils.data.dataloader.DataLoader(dd_test, batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Initializing training parameters
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Loading model to {} ...".format(device))
dd_model = CnnModel()
dd_model = CnnModel().to(device)
print("Loading parameters ...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(dd_model.parameters(), lr=learning_rate)
dd_train_dl = DeviceDataLoader(dd_trainloader, device)

print("Initializing training at {} ...".format(datetime.now().strftime("%H:%M:%S")))
# Initialize training
for epoch in range(0, epochs):
    running_loss = 0.0
    for i, data in enumerate(dd_train_dl, 0):
        # Load input
        inputs, labels = data[0].to(device), data[1].to(device)
        # Zero parameter gradients for training
        optimizer.zero_grad()
        outputs = dd_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
print("Finished training at {} ...".format(datetime.now().strftime("%H:%M:%S")))

# MODEL_PATH ="/home/alabro/dd_output/dd-b16-e15-unf-noaug.pth"
print("Saving model ... ")
MODEL_PATH = "/home/alex/A scriptie/model_saves/dd-b16-e15-unf-noaug-3clas.pth"
torch.save(dd_model.state_dict(), MODEL_PATH)
model = CnnModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH))

total = 0
correct = 0
print("Evaluating test accuracy")
with torch.no_grad():
    for data in dd_testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

print("Evaluating training acc ...")

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
# Evaluation of class accuracy
with torch.no_grad():
    for data in dd_trainloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))
