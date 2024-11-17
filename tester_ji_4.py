
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score,recall_score,precision_score
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import cv2
import optuna


data_dir = 'images'
images = os.listdir(data_dir)


test_images = pd.read_csv('CXR8/test_list.txt', header=None)


test_images

train_images = pd.read_csv('CXR8/train_val_list.txt', header=None)

train_images

labels = pd.read_csv('CXR8/Data_Entry_2017_v2020.csv')



all_labels = '|'.join(labels['Finding Labels'].unique())
all_labels = all_labels.split('|')
all_labels = list(set(all_labels))



for label in all_labels:
    labels[label] = labels['Finding Labels'].apply(lambda x: 1 if label in x else 0)


labels


tenso = torch.tensor(labels[all_labels].values).float()
data = pd.DataFrame()
data['Image Index'] = labels['Image Index']
data[all_labels] = tenso


data = data.drop(columns=['No Finding'])


all_labels

all_labels_without_no_finding = all_labels.copy()
all_labels_without_no_finding.remove('No Finding')
all_labels_without_no_finding

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

torch.cuda.empty_cache()
class CXR8Dataset(Dataset):
    def __init__(self, data, data_dir, transform=None):
        super().__init__()
        self.data = data
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data.iloc[index, 0]
        img_path = os.path.join(self.data_dir, img_name)
        img = Image.open(img_path)
        img = np.array(img)
        
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        if len(img.shape)>2:
            img = img[:,:,0]
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img)
    
        
      

        if self.transform is not None:
            img = self.transform(img)
        label = self.data.iloc[index, 1:].values
        label = np.array(label, dtype=np.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return img, label
# create a transform

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    #transforms.ToPILImage(),
   # transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
 #   transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
  #  transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.Resize(256),
    transforms.CenterCrop(256),
  #  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    #transforms.ToPILImage(),
    #transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])




    

if __name__ == '__main__':
    data = data[:50000]
 

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)



    train_dataset = CXR8Dataset(train_data, data_dir, train_transform)
    val_dataset = CXR8Dataset(val_data, data_dir, val_transform)

    # create a dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=10,  # Run single-threaded to identify issues
        pin_memory=True
      
    )

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)



    import torchvision.models as models 
    import torchvision.transforms as transforms
    from torchvision.models.resnet import ResNet, BasicBlock
    from torchvision.models.resnet import ResNet50_Weights
    from torchvision.models.densenet import DenseNet, densenet121, DenseNet121_Weights

    
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

    #model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=model.conv1.bias)
    #model.features.conv0 = nn.Conv2d(1, model.features.conv0.out_channels, kernel_size=model.features.conv0.kernel_size, stride=model.features.conv0.stride, padding=model.features.conv0.padding, bias=model.features.conv0.bias)

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 14),
       
    )

    lr = 6.469492224099435e-05
    batch_size = 47
    grad_clip = 0.9058269880647503

    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,  # Run single-threaded to identify issues
        pin_memory=True
      
    )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=10,  # Run single-threaded to identify issues
        pin_memory=True)



    

    
    # define the loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # train the model
    num_epochs = 10
    train_losses = []
    val_losses = []




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device
    print(device)

    model.to(device)
    for epoch in range(num_epochs):
        
        train_loss = 0.0
        val_loss = 0.0
        
        model.train()
        for images, labels in train_loader:
            
        
            images, labels = images.to(device), labels.to(device)


            
           
            optimizer.zero_grad()
            output = model(images)
        
            loss = criterion(output, labels.float())
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()
        
            
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels.float())
                val_loss += loss.item()
            
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))

      
        if val_loss/len(val_loader) == min(val_losses):
            torch.save(model.state_dict(), 'model.pth')

        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')

model.load_state_dict(torch.load('model.pth'))

model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        output = torch.sigmoid(output)
        output = output.cpu().detach().numpy()
        predictions.extend(output)
        actuals.extend(labels.cpu().detach().numpy())

predictions = np.array(predictions)
actuals = np.array(actuals)
predictions2 = predictions.copy()

from sklearn.metrics import roc_auc_score

# calculate the roc auc score for each label
roc_auc_scores = []
best_thresholds = []

for i in range(len(all_labels_without_no_finding)):
    
    best = 0
    best_threshold = 0
    for threshold in np.arange(0.001, 1, 0.001):
        predictions = predictions2.copy()
        predictions = np.where(predictions > threshold, 1, 0)
        roc_auc = roc_auc_score(actuals[:, i], predictions[:, i])
        if roc_auc > best:
            best = roc_auc
            best_threshold = threshold
    best_thresholds.append(best_threshold)
    roc_auc_scores.append([best, all_labels_without_no_finding[i]])

for i in range(len(all_labels_without_no_finding)):
    print(f'{all_labels_without_no_finding[i]}: {roc_auc_scores[i][0]}')




    
    


    
  

   





    




