'''
There are 60000 images, zip file of 3.1GB
mini_imagenet
 ├─labels.json
 ├─test.csv
 ├─val.csv
 ├─train.csv
 └─ images
     ├─ n0153282900000005.jpg
     ├─ n0153282900000006.jpg
     └─ ...

'''
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
root = os.environ['HOME']

class MiniImageNet(Dataset):
    def __init__(self, mode, transform):
        self.data_dir = os.path.join(root,'./Documents/datasets/mini_imagenet')
        self.csv_name = mode + '.csv'
        self.transform = transform[mode]
        assert os.path.exists(self.data_dir), 'Root dir "%s" is not found.' % (self.data_dir)
        img_dir = os.path.join(self.data_dir, './images') # ./images
        assert os.path.exists(img_dir), 'Image dir "%s" is not found' % (img_dir)
        img_label_file = os.path.join(self.data_dir, self.csv_name)
        img_label = pd.read_csv(img_label_file)

        # generate dict for converting label (n01930112) to class_value (37)
        label_mapping = pd.read_csv(os.path.join(self.data_dir, '../LOC_synset_mapping.csv'), index_col=None)
        label_mapping_dict = {}
        for i in range(label_mapping.shape[0]):
            class_value, file_name, _,_ = label_mapping.loc[i]
            label_mapping_dict[file_name] = class_value

        self.img_path_list, self.label_list = [], []
        for i in range(img_label.shape[0]):
            self.img_path_list.append(os.path.join(img_dir, img_label.loc[i,'filename']))
            label_name = img_label.loc[i,'label'] # old label in format of n01930112
            self.label_list.append(label_mapping_dict[label_name])   # new label in class value format, e.g 37

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_path_list[idx])
        assert img.mode == 'RGB', 'image "%s" is not in RGB mode' % (self.img_path_list[idx])
        label = self.label_list[idx]
        if self.transform: img = self.transform(img)
        return img, label


