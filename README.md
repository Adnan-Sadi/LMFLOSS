# LMFLOSS-A-Hybrid-Loss-For-Imbalanced-Medical-Image-Classification
This repository contains the official implementation of the LMFLoss framework from the paper [LMFLOSS: A Hybrid Loss For Imbalanced Medical Image Classification](https://arxiv.org/abs/2212.12741).

## How to Run
- Install Dependencies:
  - Please install the required libraries from the 'requirements.txt' file.

- Open the 'config.ini' file.

- Define paths: 
  - Firstly, define the path of the dataset files in the [dataset-paths] section. The path to a dataset should include 
    three folders: train, test, and val, each of which contains training, testing, and validation images. Additionally, 
    the images should be categorized into folders for each class in the dataset.
  - For example, a dataset location can be defined as, dataset_name = C:\Users\......\images
  - Next, define the path in which the trained models will be saved, in the [model-save-paths] section. There needs to be a single defined path for each dataset.
  - For example, a model save location can be defined as, dataset_name = C:\Users\......\models

- Adjusting Hyperparameters:
  - The training hyperparameters such as epochs, learning rate, weight decay and others are defined the [hyperparameters] 
    section of the 'config.ini' file. These hyperparameter can be adjusted.
  - There are also seperate sections dedicated for adjusting the hyperparameters of particular loss functions. Such as, 
    [LDAM parameters], [Focal parameters] and [LMF parameters].


- Training:
  - Before starting training first define the 'dataset', 'model', 'loss' and 'model_save_name' variables in the 
    [general] section of the config file. 
  - The 'dataset' name should be the same the defined path variable for that dataset.
  - The available keywords for 'model' and 'loss' variables are given in the config file.
  - The 'model_save_name' variable should be set to the name you wish to save your trained model as.
  - Save the 'config.ini' file once you've finished configuring the general and hyperparameter settings.
  - Finally, you can start training by running the 'train.py' file.

- Testing:
  - First, define the model filename(on which you want perform the test) in the [Load Model] section of 
    the config file.
  - The 'dataset' and 'model' should also be defined accordingly in the [general] section.
  - Finally, simply run the 'test.py' file to run the test. The test will also generate a classification report
    in the code directory.

 ## References
 - If you find our paper and code useful, please cite our paper-
 ```
 @article{sadi2022lmfloss,
  title={LMFLOSS: A Hybrid Loss For Imbalanced Medical Image Classification},
  author={Sadi, Abu Adnan and Chowdhury, Labib and Jahan, Nursrat and Rafi, Mohammad Newaz Sharif and Chowdhury, Radeya and Khan, Faisal Ahamed and Mohammed, Nabeel},
  journal={arXiv preprint arXiv:2212.12741},
  year={2022}
 }
 ```
- Also cite the papers this work is inspired from-
 ```
@inproceedings{lin2017focal,
title={Focal loss for dense object detection},
author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
booktitle={Proceedings of the IEEE international conference on computer vision},
pages={2980--2988},
year={2017}
}
```
```
@article{cao2019learning,
  title={Learning imbalanced datasets with label-distribution-aware margin loss},
  author={Cao, Kaidi and Wei, Colin and Gaidon, Adrien and Arechiga, Nikos and Ma, Tengyu},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}
```
