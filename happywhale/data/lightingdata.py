'''
Author: jianzhnie
Date: 2022-03-29 11:18:22
LastEditTime: 2022-03-30 17:16:33
LastEditors: jianzhnie
Description:

'''

from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader

from happywhale.data.happywhale import HappyWhaleDataset


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_file: str,
        test_csv_file: str,
        val_fold: float,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_df = pd.read_csv(train_csv_file)
        self.test_df = pd.read_csv(test_csv_file)

        self.transform = create_transform(
            input_size=(self.hparams.image_size, self.hparams.image_size),
            crop_pct=1.0,
        )

    def setup(self, stage: Optional[str] = None):
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

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self,
                    dataset: HappyWhaleDataset,
                    train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=train,
        )
