
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
import torchvision.models as models 
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.densenet import DenseNet, densenet121, DenseNet121_Weights
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

    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 50)
    grad_clip = trial.suggest_float('grad_clip', 0.1, 1.0)
    scheduler_name = trial.suggest_categorical(
        'scheduler_name', ['None', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'StepLR', 'ExponentialLR']
    )
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)

    
    train_transform_random_resize = transforms.Compose([
    transforms.RandomHorizontalFlip(),
 
    transforms.RandomRotation(7),
    transforms.RandomResizedCrop(
        size=(224, 224),  
        scale=(0.08, 1.0),
        ratio=(3/4, 4/3)
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])



    
    train_transform = train_transform_random_resize

    train_dataset = CXR8Dataset(train_data, data_dir, train_transform)
    val_dataset = CXR8Dataset(val_data, data_dir, val_transform)
    test_dataset = CXR8Dataset(test_data, data_dir, val_transform)
    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=10,  
            pin_memory=True
        
        )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,num_workers=10, pin_memory=True,
                                prefetch_factor=2 
                            
                            )
    
    

    
    # train the model
    num_epochs = 20
    train_losses = []
    val_losses = []

    patience = 3
    counter = 0



    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

      

    model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 14)
    
        )

        # define the loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

    if scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    elif scheduler_name == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, last_epoch=-1, verbose=False)
    elif scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif scheduler_name == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1)
    else:
        scheduler = None


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

                if scheduler_name == 'OneCycleLR':
                    scheduler.step()

               
            
                train_loss += loss.item()
            
                
            model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    output = model(images)
                    loss = criterion(output, labels.float())
                    val_loss += loss.item()
                

            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            elif scheduler_name == 'None' or scheduler_name == 'OneCycleLR':
                pass
            else:
                scheduler.step()
            train_losses.append(train_loss/len(train_loader))
            val_losses.append(val_loss/len(val_loader))

            
            if val_loss/len(val_loader) == min(val_losses):
                
                counter = 0
            
            trial.report(val_loss/len(val_loader), epoch)
            if trial.should_prune():
                print('Pruned')
                break

            counter += 1

            if counter == patience:
                print(f'Last Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')  
                break

        
        

            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')  
    return min(val_losses) 

   



   

   

if __name__ == '__main__':

    # place the images in train_images in the train set and the images in test_images in the test set
    train_val_data = data[data['Image Index'].isin(train_images[0].values)]
    test_data = data[data['Image Index'].isin(test_images[0].values)]

    patients_ids = pd.read_csv('CXR8/Data_Entry_2017_v2020.csv')
    patients_ids = patients_ids[['Image Index', 'Patient ID']]

    train_val_data = pd.merge(train_val_data, patients_ids, on='Image Index')



    unique_patient_ids = train_val_data['Patient ID'].unique()

    # Split patient IDs into training and validation sets
    train_patient_ids, val_patient_ids = train_test_split(
        unique_patient_ids, 
        test_size=0.1, 
        random_state=42
    )

    # Create train and validation data based on the patient ID split
    train_data = train_val_data[train_val_data['Patient ID'].isin(train_patient_ids)]
    val_data = train_val_data[train_val_data['Patient ID'].isin(val_patient_ids)]

    train_data = train_data.drop(columns=['Patient ID'])
    val_data = val_data.drop(columns=['Patient ID'])





  





  



    
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(), study_name='optuna_local_ResNEt_1', storage='sqlite:///ChestX.db', load_if_exists=True)

    try:
        study.optimize(objective, n_trials=100)
    except KeyboardInterrupt:
        print('Interrupted by keyboard')
        print(study.best_params)
        print(study.best_value)

        

        

    




