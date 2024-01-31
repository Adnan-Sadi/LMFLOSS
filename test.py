# -*- coding: utf-8 -*-
"""
@author: Adnan-Sadi
"""

import torch
import pandas as pd
import os
import numpy as np
from torchvision import transforms,datasets
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import random
import configparser

from utils import choose_dataset
from dl_models import get_efficientnetv2, get_resnet50, get_densenet121

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
    batch_size = int(config.get('hyperparameters', 'batch_size'))
    num_workers = int(config.get('hyperparameters', 'num_workers'))
    print(data_dir, model_save_dir)
    #%%
    h = int(config.get('hyperparameters', 'image_height'))
    w = int(config.get('hyperparameters', 'image_width'))

    test_transforms = transforms.Compose([
                      transforms.Resize((h,w)),
                      transforms.ToTensor(),
                      ])
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_transforms)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers)
    test_set_size = len(test_set)
    class_names = test_set.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #%%
    model_name = config.get('general', 'model')
    out_ftrs = len(class_names)
    freeze_wts = config.getboolean('hyperparameters', 'freeze_CNN_weights')

    if(model_name == 'efficientnetv2'):
        model = get_efficientnetv2(out_ftrs, freeze_wts)
    elif(model_name == 'densenet121'):
        model = get_densenet121(out_ftrs, freeze_wts)
    else:
        model = get_resnet50(out_ftrs, freeze_wts)

    model.to(device)

    filename = config.get('Load Model', 'Filename')
    model.load_state_dict(torch.load(model_save_dir + "\\" + filename+ ".pth"))
    model = model.to(device)
    #%%
    num_correct = 0
    num_samples = test_set_size
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds = torch.cat((all_preds, preds))
            all_labels = torch.cat((all_labels,labels.data))
            num_correct += torch.sum(preds == labels.data)
            

        test_acc = num_correct.double() / num_samples
            
        test_precision = precision_score(all_labels.cpu(), all_preds.cpu(), average='macro', zero_division=0)
        test_recall = recall_score(all_labels.cpu(), all_preds.cpu(), average='macro', zero_division=0)
        test_f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average='macro')
        test_f1_wt = f1_score(all_labels.cpu(), all_preds.cpu(), average='weighted')
        
        print(f' Test Acc: {test_acc:.4f} F1: {test_f1:.4f} F1_Weighted: {test_f1_wt:.4f} Precision: {test_precision:.4f} Recall: {test_recall:.4f}')

    model.train()

    all_labels = all_labels.tolist()
    all_labels = [int(i) for i in all_labels]
    all_preds = all_preds.tolist()
    all_preds = [int(i) for i in all_preds]

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    print(df)

    df.to_csv('report_'+ filename +'.csv')
    

if __name__ == '__main__':
    main()