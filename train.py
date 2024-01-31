# -*- coding: utf-8 -*-
"""
@author: Adnan-Sadi
"""
from __future__ import print_function, division
from __future__ import print_function

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch.utils.tensorboard import SummaryWriter
import configparser
import random

from loss_functions import LDAMLoss,focal_loss, LMFLoss, bwcce_loss
from utils import choose_dataset, get_img_num_per_cls, get_mlist, parameter_count
from dl_models import get_efficientnetv2, get_resnet50, get_densenet121

cudnn.benchmark = True
plt.ion()   # interactive mode

random.seed(23)
torch.manual_seed(23)
torch.cuda.manual_seed(23)
np.random.seed(23)
#%%

def main():
    config = configparser.ConfigParser()
    config.read_file(open(r'config.ini'))

    dataset_name = config.get('general', 'dataset')
    data_dir, model_save_dir = choose_dataset(dataset_name)
    #%%
    batch_size = int(config.get('hyperparameters', 'batch_size'))
    num_workers = int(config.get('hyperparameters', 'num_workers'))
    h = int(config.get('hyperparameters', 'image_height'))
    w = int(config.get('hyperparameters', 'image_width'))

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((h,w)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((h,w)),
            transforms.ToTensor(),
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers)
                  for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #%%
    print(image_datasets)
    print('\n', dataloaders)
    print('\n', dataset_sizes)
    print('\n', class_names)
    print('\n', device)
    #%%
    train_dir = os.path.join(data_dir, 'train')
    img_per_cls = get_img_num_per_cls(train_dir)
    m_list_wts = get_mlist(img_per_cls)
    #%%
    model_name = config.get('general', 'model')
    out_ftrs = len(class_names)
    freeze_wts = config.getboolean('hyperparameters', 'freeze_CNN_weights')

    if(model_name == 'efficientnetv2'):
        model = get_efficientnetv2(out_ftrs, freeze_wts)
    elif(model_name == 'resnet50'):
        model = get_resnet50(out_ftrs, freeze_wts)
    elif(model_name == 'densenet121'):
        model = get_densenet121(out_ftrs, freeze_wts)

    model.to(device)
    print("Model Parameters- ")
    parameter_count(model)
    #%%
    lr = float(config.get('hyperparameters', 'lr'))
    weight_decay  = float(config.get('hyperparameters', 'weight_decay'))
    scheduler_step = int(config.get('hyperparameters', 'scheduler_step'))
    scheduler_gamma = float(config.get('hyperparameters', 'scheduler_gamma'))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    loss_name = config.get('general', 'loss')
    #%%
    max_m = float(config.get('LDAM parameters', 'max_m'))
    s = int(config.get('LDAM parameters', 's'))
    gamma = float(config.get('Focal parameters', 'gamma'))

    if (loss_name == 'LDAM'):
        criterion = LDAMLoss(cls_num_list=m_list_wts, max_m=max_m,
                             weight=torch.Tensor(m_list_wts).to(device), s=s)
        
    elif(loss_name == 'FOCAL'):
        criterion = focal_loss(device=device, alpha=torch.Tensor(m_list_wts).to(device)
                               , gamma = gamma)
        
    elif(loss_name == 'LMF'):
        alpha = float(config.get('LMF parameters', 'alpha'))
        beta = float(config.get('LMF parameters', 'beta'))
        
        criterion1 = LDAMLoss(cls_num_list=m_list_wts, max_m=max_m,
                             weight=torch.Tensor(m_list_wts).to(device), s=s)
        criterion2 = focal_loss(device=device, alpha=torch.Tensor(m_list_wts).to(device)
                               , gamma = gamma)
        criterion = LMFLoss(criterion1, criterion2, alpha= alpha, beta=beta)
        print("alpha: ", criterion.alpha, " and Beta: ", criterion.beta)
        
    elif(loss_name == 'CrossEntropy'):
        criterion = torch.nn.CrossEntropyLoss()
    
    elif(loss_name == 'BWCCE'):
        criterion = bwcce_loss(img_per_cls, device)
        
    num_epochs = int(config.get('hyperparameters', 'epochs'))
    print("Loss Function: ", criterion)
    #%%
    filename = config.get('general', 'model_save_name')
    
    # Training Model
    since = time.time()
    writer = SummaryWriter(log_dir='runs/' + filename) 

    best_model_wts = copy.deepcopy(model.state_dict()) #best weights
    best_acc = 0.0
    best_train_acc= 0.0
    best_train_f1= 0.0
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            all_preds = torch.tensor([]).to(device)
            all_labels = torch.tensor([]).to(device)
                
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                    
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                        
                    loss = criterion(outputs, labels)
                
                    #get all preds and labels for calculating f1,precision,recall
                    all_preds = torch.cat((all_preds, preds))
                    all_labels = torch.cat((all_labels,labels.data))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                    
                    
            if phase == 'train':
                print("lr:", optimizer.param_groups[0]['lr'])
                scheduler.step()  #comment this line to turn off scheduler

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
            epoch_precision = precision_score(all_labels.cpu(), all_preds.cpu(), average='macro', zero_division=0)
            epoch_recall = recall_score(all_labels.cpu(), all_preds.cpu(), average='macro', zero_division=0)
            epoch_f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average='macro')
                
            writer.add_scalar("Loss/"+ phase, epoch_loss, epoch)
            writer.add_scalar("Acc/"+ phase, epoch_acc, epoch)
            writer.add_scalar("F1/"+ phase, epoch_f1, epoch)
                
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f}')
                
            if phase == 'train' and epoch_acc > best_train_acc:
                best_train_acc = epoch_acc
                
            if phase == 'train' and epoch_f1 > best_train_f1:
                best_train_f1 = epoch_f1

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                    
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                    

        print()
        
    writer.flush()
    writer.close()
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    print(f'Best train Acc: {best_train_acc:4f}')
    print(f'Best Train F1 Score: {best_train_f1:4f}')
    print(f'Best Val F1 Score: {best_f1:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
   
    save_file = filename+".pth"
    torch.save(model.state_dict(), os.path.join(model_save_dir,save_file))  


if __name__ == '__main__':
    main()


