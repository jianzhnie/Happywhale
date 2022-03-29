'''
Author: jianzhnie
Date: 2022-03-29 11:34:27
LastEditTime: 2022-03-29 11:40:13
LastEditors: jianzhnie
Description:

'''
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from .data.lightingdata import LitDataModule
from .models.lighting_model import LitModule


def get_image_path(id, dir):
    return os.path.join(dir, id)


if __name__ == '__main__':
    train_dir = '/home/easyits/road-risk-identification/kaggle/dataset2/cropped_train_images/cropped_train_images/'
    test_dir = '/home/easyits/road-risk-identification/kaggle/dataset2/cropped_test_images/cropped_test_images'
    train_csv_path = '/home/easyits/road-risk-identification/kaggle/dataset2/train2.csv'
    test_csv_path = '/home/easyits/road-risk-identification/kaggle/dataset2/test2.csv'
    encoder_classes_path = '/home/easyits/road-risk-identification/kaggle/dataset2/encoder_classes.npy'
    train_csv_encoded_folded_path = '/home/easyits/road-risk-identification/kaggle/dataset2/train_encoded_folded.csv'
    # 五折交叉验证
    n_splits = 5
    # 权重保存路径
    checkpoints_dir = '/home/easyits/road-risk-identification/kaggle/checkpoints'
    # 是否debug
    debug = False

    sample_submission_csv_path = '/home/easyits/road-risk-identification/kaggle/dataset2/sample_submission.csv'
    # 最终提交csv文件
    submission_csv_path = './submission.csv'

    #  ================Train DataFrame=================
    train_df = pd.read_csv(train_csv_path)
    train_df['image_path'] = train_df['image'].apply(get_image_path,
                                                     dir=train_dir)

    # 给类别编码
    encoder = LabelEncoder()
    train_df['individual_id'] = encoder.fit_transform(
        train_df['individual_id'])
    np.save(encoder_classes_path, encoder.classes_)

    # 五折交叉验证
    skf = StratifiedKFold(n_splits=n_splits)
    for fold, (_, val_) in enumerate(
            skf.split(X=train_df, y=train_df.individual_id)):
        train_df.loc[val_, 'kfold'] = fold

    train_df.to_csv(train_csv_encoded_folded_path, index=False)
    print(train_df.head())

    # ================Test DataFrame================
    test_df = pd.read_csv(sample_submission_csv_path)
    test_df['image_path'] = test_df['image'].apply(get_image_path,
                                                   dir=test_dir)
    test_df.drop(columns=['predictions'], inplace=True)

    test_df['individual_id'] = 0
    test_df.to_csv(test_csv_path, index=False)
    test_df.head()

    def train(train_csv_encoded_folded=str(train_csv_encoded_folded_path),
              test_csv=str(test_csv_path),
              val_fold=0.0,
              image_size=256,
              batch_size=64,
              num_workers=4,
              model_name='tf_efficientnet_b0',
              pretrained=True,
              drop_rate=0.0,
              embedding_size=512,
              num_classes=15587,
              arc_s=30.0,
              arc_m=0.5,
              arc_easy_margin=False,
              arc_ls_eps=0.0,
              optimizer='adam',
              learning_rate=3e-4,
              weight_decay=1e-6,
              checkpoints_dir=str(checkpoints_dir),
              accumulate_grad_batches=1,
              auto_lr_find=False,
              auto_scale_batch_size=False,
              fast_dev_run=False,
              gpus=4,
              max_epochs=100,
              precision=16,
              stochastic_weight_avg=True):

        pl.seed_everything(42)

        # 定义数据集
        datamodule = LitDataModule(
            train_csv_encoded_folded=train_csv_encoded_folded,
            test_csv=test_csv,
            val_fold=val_fold,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        datamodule.setup()
        len_train_dl = len(datamodule.train_dataloader())

        # 定义模型
        module = LitModule(model_name=model_name,
                           pretrained=pretrained,
                           drop_rate=drop_rate,
                           embedding_size=embedding_size,
                           num_classes=num_classes,
                           arc_s=arc_s,
                           arc_m=arc_m,
                           arc_easy_margin=arc_easy_margin,
                           arc_ls_eps=arc_ls_eps,
                           optimizer=optimizer,
                           learning_rate=learning_rate,
                           weight_decay=weight_decay,
                           len_train_dl=len_train_dl,
                           epochs=max_epochs)

        # 初始化ModelCheckpoint回调，并设置要监控的量。
        #  monitor：需要监控的量，string类型。
        # 例如'val_loss'（在training_step() or validation_step()函数中通过self.log('val_loss', loss)进行标记）；
        # 默认为None，只保存最后一个epoch的模型参数,
        model_checkpoint = ModelCheckpoint(
            checkpoints_dir,
            filename=f'{model_name}_{image_size}',
            monitor='val_loss',
            save_top_k=5)

        # 定义trainer
        trainer = pl.Trainer(
            accumulate_grad_batches=accumulate_grad_batches,  # 每k次batches累计一次梯度
            auto_lr_find=auto_lr_find,
            auto_scale_batch_size=auto_scale_batch_size,
            benchmark=True,
            callbacks=[model_checkpoint],  # 添加回调函数或回调函数列表
            deterministic=True,
            fast_dev_run=fast_dev_run,
            gpus=gpus,  # 使用的gpu数量(int)或gpu节点列表(list或str)
            max_epochs=2 if debug else max_epochs,  # 最多训练轮数
            precision=precision,
            stochastic_weight_avg=stochastic_weight_avg,
            limit_train_batches=0.1
            if debug else 1.0,  # 使用训练/测试/验证/预测数据的百分比。如果数据过多,或正在调试可以使用。
            limit_val_batches=0.1 if debug else 1.0,
        )

        # Trainer.tune()对模型超参数进行调整
        trainer.tune(module, datamodule=datamodule)

        # 开始训练
        # Trainer.fit() 参数详解
        # model->LightningModule实例;
        # train_dataloaders->训练数据加载器
        # val_dataloaders->验证数据加载器
        # ckpt_path->ckpt文件路径(从这里文件恢复训练)
        # datamodule->LightningDataModule实例
        trainer.fit(module, datamodule=datamodule)

    model_name = 'swin_large_patch4_window12_384_in22k'  # "convnext_small"
    image_size = 384
    batch_size = 64

    train(model_name=model_name, image_size=image_size, batch_size=batch_size)
