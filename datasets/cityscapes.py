'''
1. download zip files via:
 https://www.cityscapes-dataset.com/dataset-overview/
 sign up and log in before downloading
 gtFine_trainvaltest.zip (241MB) [md5], leftImg8bit_trainvaltest.zip (11GB) [md5]

2. manually unzip or run codes to unzip files, as:

from torchvision.datasets import Cityscapes as cityscapes  # will unzip the file when run at first time
import os
root ='/home/royliu/Documents/datasets/'
data_dir ='cityscapes'
data_dir = os.path.join(root, data_dir)
dataset = cityscapes(data_dir, split = 'train', mode ='fine', target_type = 'semantic')

will get data in below structure

├── gtFine
│   ├── train
│   │   ├── aachen
│   │   ├── bochum
│   │   └── ......
│   └── val
│       └── frankfurt
└── leftImg8bit
    ├── train
    │   ├── aachen
    │   ├── bochum
    │   └── ......
    └── val
        └── frankfurt


output:  4 dimentions ndarray of: mask, image
'''

import cv2
import numpy as np
import os
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
from PIL import Image
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', metavar='data_dir', default='./Documents/datasets/cityscapes')
parser.add_argument('--split', metavar='split', default='train', help='"train" or "val" or "test".')
root = os.environ['HOME']

class DataGenerator(Cityscapes):  ## varible names in Cityscapes Class are: images, targets, split...
    def __init__(self,  *args, **kwargs):
        super().__init__( *args, **kwargs, target_type="semantic")
        # if not type(image_size) == tuple: image_size = (image_size,image_size)
        self.image_size = (224,224) ## Roy: 图片尺寸改这里
        self.semantic_target_type_index = [i for i, t in enumerate(self.target_type) if t == "semantic"][0]
        self.colormap = self._generate_colormap()


        # self.transform = self._transform()

    def _transform(self, image, mask):
        transform_img = transforms.Compose([
                                # transforms.Resize((256, 512)),
                                transforms.Resize(self.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        transform_mask = transforms.Compose([
                                # transforms.Resize((256, 512))])
                                transforms.Resize(self.image_size)])

        return transform_img(image), transform_mask(mask)


    def _generate_colormap(self):
        colormap = {}
        for class_ in self.classes:
            if class_.train_id in (-1, 255):
                continue
            colormap[class_.train_id] = class_.id
        return colormap  # return a diction

    def _convert_to_segmentation_mask(self, mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(self.colormap)), dtype=np.float32)
        for label_index, label in self.colormap.items():
            segmentation_mask[:, :, label_index] = (mask == label).astype(float)
        return segmentation_mask

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        mask = cv2.imread(self.targets[index][self.semantic_target_type_index], cv2.IMREAD_UNCHANGED)
        mask = self._convert_to_segmentation_mask(mask)
        mask = mask.transpose(2, 0, 1)  # transfer shape to class#, h, w
        mask = torch.from_numpy(mask)

        image, mask =self._transform(image, mask)
        return image, mask

def cv2_to_pil(img_cv): # convert cv2 to PIL format
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img_pil

class CityTransform:  # to transform image and mask
    def __call__(self, image, mask, image_size=(256,512)):
        # image = cv_to_pil(image)  # to satisefy transforms.RandomResizedCrop's input requirement in shape of (...,h,w)
        transform_img = transforms.Compose([
                                # transforms.RandomResizedCrop((256, 512)),
                                transforms.RandomResizedCrop(image_size),
                                # transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        transform_mask = transforms.Compose([
                                # transforms.RandomResizedCrop((256, 512))])
                                transforms.RandomResizedCrop(image_size)])
        return transform_img(image), transform_mask(mask)

if __name__=='__main__':

    args = parser.parse_args()
    data_dir = os.path.join(root, args.data_dir)
    assert os.path.exists(data_dir), 'Root of dataset is incorrect or miss.'

    # dataset_train = DataGenerator( args.root,  split = 'train') # default: mode='fine', target_type= 'sementic, split: train, test or val if mode=”fine” otherwise train, train_extra or val
    # img_tensor, sgm = dataset_train[0]

    dataset_test = DataGenerator(data_dir, split='test')
    img_tensor, sgm = dataset_test[0]
    test_loader = torch.utils.data.DataLoader(dataset_test, \
                                              batch_size= 1, num_workers=1)
    print(img_tensor.shape, sgm.shape)
    # print(img_tensor)


