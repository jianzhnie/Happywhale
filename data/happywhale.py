'''
Author: jianzhnie
Date: 2022-03-29 11:11:54
LastEditTime: 2022-03-29 11:43:19
LastEditors: jianzhnie
Description:

'''
import torch
from PIL import Image
from torch.utils.data import Dataset


class HappyWhaleDataset(Dataset):
    def __init__(self, df, train=True, transform=None):
        self.df = df
        self.transform = transform
        self.image_names = self.df['image'].values
        self.image_paths = self.df['image_path'].values
        if train:
            self.targets = self.df['individual_id'].values
        else:
            self.targets = None

    def __getitem__(self, index):
        # 图片名字
        image_name = self.image_names[index]
        # 图片路径
        image_path = self.image_paths[index]

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        if self.targets:
            target = self.targets[index]
        else:
            target = None

        target = torch.tensor(target, dtype=torch.long)

        return {'image_name': image_name, 'image': image, 'target': target}

    def __len__(self):
        return len(self.df)
