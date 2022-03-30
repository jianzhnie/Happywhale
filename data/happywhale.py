'''
Author: jianzhnie
Date: 2022-03-29 11:11:54
LastEditTime: 2022-03-30 15:02:02
LastEditors: jianzhnie
Description:

'''

from typing import Callable, Dict, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class HappyWhaleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None):
        self.df = df
        self.transform = transform

        self.image_names = self.df['image'].values
        self.image_paths = self.df['image_path'].values
        self.targets = self.df['individual_id'].values

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # 图片名字
        image_name = self.image_names[index]
        # 图片路径
        image_path = self.image_paths[index]

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        target = self.targets[index]
        target = torch.tensor(target, dtype=torch.long)

        return {'image_name': image_name, 'image': image, 'target': target}

    def __len__(self) -> int:
        return len(self.df)
