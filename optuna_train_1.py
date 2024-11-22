
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



import optuna
def objective(trial):

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = 32
    grad_clip = trial.suggest_float('grad_clip', 0.1, 1.0)
    random_rotation = trial.suggest_int('random_rotation', 0, 180)
    collor_jitter = trial.suggest_float('collor_jitter', 0, 0.5)
    scheduler_name = trial.suggest_categorical('scheduler_name', ['StepLR', 'ReduceLROnPlateau', 'OneCycleLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts'])



    train_transform = transforms.Compose([
    #transforms.ToPILImage(),
   # transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
 #   transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
  #  transforms.RandomVerticalFlip(),
    transforms.RandomRotation(random_rotation),
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ColorJitter(brightness=collor_jitter, contrast=collor_jitter, saturation=collor_jitter, hue=collor_jitter),
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



    train_dataset = CXR8Dataset(train_data, data_dir, train_transform)
    val_dataset = CXR8Dataset(val_data, data_dir, val_transform)
    test_dataset = CXR8Dataset(test_data, data_dir, val_transform)


    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

    #model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=model.conv1.bias)
    #model.features.conv0 = nn.Conv2d(1, model.features.conv0.out_channels, kernel_size=model.features.conv0.kernel_size, stride=model.features.conv0.stride, padding=model.features.conv0.padding, bias=model.features.conv0.bias)

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 14),
     #   nn.ReLU(),
      #  nn.Dropout(0.5),
      #  nn.Linear(1024, 512),
      #  nn.ReLU(),
      #  nn.Dropout(0.5),
      #  nn.Linear(512, 14),
    )

    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,  # Run single-threaded to identify issues
        pin_memory=True
      
    )

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,num_workers=10, pin_memory=True,
                              prefetch_factor=2 
                           
                           )
    

    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=10, pin_memory=True,
                              prefetch_factor=2)
    



    

    
    # define the loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    if scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    elif scheduler_name == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=trial.suggest_float('max_lr', 1e-5, 1e-1, log=True), 
                                                steps_per_epoch=len(train_loader), epochs=num_epochs)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trial.suggest_int('T_max', 1, 50), 
                                                        eta_min=trial.suggest_float('eta_min', 0, 1e-3))
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=trial.suggest_int('T_0', 1, 50), 
                                                                T_mult=trial.suggest_int('T_mult', 1, 5), 
                                                                eta_min=trial.suggest_float('eta_min', 0, 1e-3))

    

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

            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            elif scheduler_name == 'None':
                pass
            else:
                scheduler.step()
        
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

      
      

        trial.report(val_loss/len(val_loader), epoch)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')  
        if trial.should_prune():
            raise optuna.TrialPruned()
    return min(val_losses)

        
       


   

   

if __name__ == '__main__':

    # place the images in train_images in the train set and the images in test_images in the test set
    train_val_data = data[data['Image Index'].isin(train_images[0].values)]
    test_data = data[data['Image Index'].isin(test_images[0].values)]
    train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)






  


    import torchvision.models as models 
    import torchvision.transforms as transforms
    from torchvision.models.resnet import ResNet, BasicBlock
    from torchvision.models.resnet import ResNet50_Weights
    from torchvision.models.densenet import DenseNet, densenet121, DenseNet121_Weights

    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(), study_name='optuna_train_4', storage='sqlite:///ChestX.db', load_if_exists=True)

    try:
        study.optimize(objective, n_trials=100)
    except KeyboardInterrupt:
        print('Interrupted by keyboard')
        print(study.best_params)

    
 
    




