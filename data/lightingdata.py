'''
Author: jianzhnie
Date: 2022-03-29 11:18:22
LastEditTime: 2022-03-29 11:22:44
LastEditors: jianzhnie
Description:

'''

import pandas as pd
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from .happywhale import HappyWhaleDataset


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_encoded_folded,
        test_csv,
        val_fold,
        image_size,
        batch_size,
        num_workers,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_df = pd.read_csv(train_csv_encoded_folded)
        self.test_df = pd.read_csv(test_csv)

        self.transform = create_transform(input_size=(self.hparams.image_size,
                                                      self.hparams.image_size),
                                          crop_pct=1.0)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Split train df using fold
            train_df = self.train_df[
                self.train_df.kfold != self.hparams.val_fold].reset_index(
                    drop=True)
            val_df = self.train_df[self.train_df.kfold == self.hparams.
                                   val_fold].reset_index(drop=True)

            self.train_dataset = HappyWhaleDataset(train_df,
                                                   transform=self.transform)
            self.val_dataset = HappyWhaleDataset(val_df,
                                                 transform=self.transform)

        if stage == 'test' or stage is None:
            self.test_dataset = HappyWhaleDataset(self.test_df,
                                                  transform=self.transform)

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset, train=False):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=train,
        )
