'''
download: https://www.kaggle.com/datasets/862ae86edba271c39f76d0b530edeb55076b4b82b971160637210900747c44b1/download?datasetVersionNumber=1
    origin from: https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1

dataset strcutre:

│   ├───caltech-101
│   │   ├───accordion
│   │   ├───airplanes
│   │   ├───anchor
│   │   ├───ant
│   │   ├───BACKGROUND_Google
│   │   ├───barrel
│   │   ├───bass
│   │   ├───beaver
│   │   ├───binocular
│   │   ├───bonsai
│   │   ├───brain
│   │   ├───brontosaurus
...

output:  pytorch dataloader
'''


import numpy as np
import os
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import random
import sys
sys.path.append("../src")
from util import dataset_split

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', metavar='data-dir', default='./Documents/datasets/caltech-101')
parser.add_argument('--split', metavar='split', default='train', help='"train" or "val" or "test".')
parser.add_argument('--split-rate', metavar='split-rate', default= (0.6,0.2,0.2), help='a tuple: e.g. (0.6,0.2,0.2)')
root = os.environ['HOME']

def encoder_label(labels_name_list):
    label_names = list(dict.fromkeys(labels_name_list)) #unique
    encoder ={}
    label_encoder_list = []
    for i, name in enumerate(label_names): encoder[name]=i
    for label_name in labels_name_list:
        label_encoder_list.append(int(encoder[label_name]))
    return label_encoder_list



class GenerateDataset(Dataset):
    def __init__(self,  image_size, mode):
        args = parser.parse_args()
        self.image_size = image_size
        self.mode = mode
        split_rate = args.split_rate
        image_dir = os.path.join(root, args.data_dir)
        image_folders = os.listdir(image_dir)
        self.img_path_list, self.label_list = [], []
        for sub_folder in (image_folders):
            if sub_folder == 'BACKGROUND_Google': continue  ## bypass this folder
            sub_folder_path = os.path.join(image_dir, sub_folder)

            for image in (os.listdir(sub_folder_path)):
                image_path = os.path.join(sub_folder_path, image)
                self.img_path_list.append(image_path)
                self.label_list.append(sub_folder)
        self.label_list = encoder_label(self.label_list)  # encoder label: label in name to label in value

        ### split train, val, test , e.g. in rate of 0.6, 0.2, 0.2
        self.img_path_list_trn, self.img_path_list_val, self.img_path_list_test = \
            dataset_split(self.img_path_list, split_rate = split_rate, seed=42)  # generate dataset speration index
        self.label_list_trn, self.label_list_val, self.label_list_test = \
            dataset_split(self.label_list, split_rate = split_rate, seed=42)

    def _transform(self, image_pil):
        transform = transforms.Compose([
                                transforms.Resize(self.image_size),
                                transforms.ToTensor(),
        ])
        return transform(image_pil)

    def __getitem__(self, index):
        if self.mode == 'train':
            image_path, label = self.img_path_list_trn, self.label_list_trn
        elif self.mode == 'val':
            image_path, label = self.img_path_list_val, self.label_list_val
        else: # test
            image_path, label = self.img_path_list_test, self.label_list_test

        # image_path, label = self.img_path_list[index], self.label_list[index]
        image = Image.open(image_path[index])
        if len(np.array(image).shape) < 3:  # gray image
            image = image.convert("RGB")  # if gray imange , change to RGB (3 channels)
        image = self._transform(image)
        return image, label[index]

    def __len__(self):
        if self.mode == 'train':
            length = len(self.img_path_list_trn)
        elif self.mode == 'val':
            length = len(self.img_path_list_val)
        else:  ## test
            length = len(self.img_path_list_test)
        return length

def main():
    dataset = GenerateDataset(image_size = (224,224), mode = 'train')
    data_loader = DataLoader(dataset, batch_size= 32, num_workers= 1, shuffle=False)
    print('Dataload length:', len(data_loader))

if __name__ == '__main__':
    main()
