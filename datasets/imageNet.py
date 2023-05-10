'''
download:
 https://image-net.org/challenges/LSVRC/2012/index.php
 or, https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data

ILSVRC2012_img_train.tar , 138GB.
ILSVRC2012_img_val.tar , 6.3GB.
Training bounding box annotations (Task 1 & 2 only). 20MB.

For extracting training and  val dataset after downloading, run below script perspectively (under the same folder of the tar file):

        mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
        tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
        find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
        cd ..
        #
        # Extract the validation data and move images to subfolders:
        mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
        wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash


train/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├── n01440764_10029.JPEG
│   └── ...
├── n01443537
│   ├── n01443537_10007.JPEG
│   ├── n01443537_10014.JPEG
│   ├── n01443537_10025.JPEG
│   └── ...
├── ...
└── ...

val/
├── n01440764
│   ├── ILSVRC2012_val_00000946.JPEG
│   ├── ILSVRC2012_val_00001684.JPEG
│   └── ...
├── n01443537
│   ├── ILSVRC2012_val_00001269.JPEG
│   ├── ILSVRC2012_val_00002327.JPEG
│   ├── ILSVRC2012_val_00003510.JPEG
│   └── ...
├── ...
└── ...
val/
├── n01440764
'''

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from PIL import Image
import numpy as np

parser =  argparse.ArgumentParser()
parser.add_argument('--data-dir', metavar = 'data-dir', default= './Documents/datasets/imagenet') #/media/lab/Data/datasets/imagenet
root = os.environ['HOME']

def normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


def train_dataset(data_dir, image_size):
    # train_dir = os.path.join(data_dir, 'ILSVRC2012_img_train_t3')
    train_dir = os.path.join(data_dir, 'ILSVRC2012_img_train')

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform()
    ])

    train_dataset = datasets.ImageFolder(
        train_dir,
        train_transforms
    )

    return train_dataset


def val_dataset(data_dir, image_size):
    val_dir = os.path.join(data_dir, 'ILSVRC2012_img_val')

    val_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.ToTensor(),
        normalize_transform()
    ])

    val_dataset = datasets.ImageFolder(
        val_dir,
        val_transforms
    )

    return val_dataset

def data_loader(batch_size=256, workers=2, pin_memory=True, image_size=(224,224)):
    args = parser.parse_args()
    data_dir = os.path.join(os.environ['HOME'],args.data_dir)
    train_ds = train_dataset(data_dir, image_size)
    val_ds = val_dataset(data_dir, image_size)


    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader

class test_dataset(Dataset):
    def __init__(self, data_dir,image_size=(224,224)):
        # self.test_dir= os.path.join(data_dir, 'test')
        self.test_dir = os.path.join(data_dir, 'ILSVRC2012_img_test_v10102019')
        self.images = []
        self.image_size = image_size
        for file_name in os.listdir(self.test_dir):
            self.images.append(os.path.join(self.test_dir, file_name))

    def _transform(self, image):
        transform =transforms.Compose([
        transforms.RandomResizedCrop(self.image_size),
        transforms.ToTensor(),
        normalize_transform()
         ])
        return transform(image)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        if len(np.array(image).shape) <3: # if it is gray image
            image = image.convert("RGB")  # if gray imange , change to RGB (3 channels)
        return self._transform(image), -1 # all none labeled image to be labeled as -1

    def __len__(self):
        return len(self.images)

def test_loader(batch_size=16, workers=1, pin_memory=True, image_size=(224,224)):
    args = parser.parse_args()
    data_dir = args.data_dir
    test_ds = test_dataset(data_dir, image_size)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    return test_loader

if __name__ =='__main__':

    batch_size= 32
    workers =2
    image_size = 224
    train_loader, val_loader =  data_loader(batch_size =  batch_size, workers = workers, image_size = image_size)
    dataload_test = test_loader(batch_size =  batch_size, workers = workers, image_size=image_size)
    print('Train folders:', len(train_loader))
    # test_ds = test_dataset( '/data/datasets/imagenet')
    # print(test_ds[0])

