'''
Author: jianzhnie
Date: 2022-03-29 11:11:54
LastEditTime: 2022-03-29 11:42:14
LastEditors: jianzhnie
Description:

'''
# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_data_transforms(image_size):

    data_transforms = {
        'train':
        A.Compose([
            A.Resize(image_size, image_size),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2,
                                 sat_shift_limit=0.2,
                                 val_shift_limit=0.2,
                                 p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                       contrast_limit=(-0.1, 0.1),
                                       p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0),
            ToTensorV2()
        ],
                  p=1.),
        'valid':
        A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0),
            ToTensorV2()
        ],
                  p=1.)
    }
    return data_transforms
