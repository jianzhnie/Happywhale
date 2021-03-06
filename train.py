'''
Author: jianzhnie
Date: 2022-03-29 11:34:27
LastEditTime: 2022-03-30 17:53:27
LastEditors: jianzhnie
Description:

'''
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from happywhale.data.lightingdata import LitDataModule
from happywhale.models.lighting_model import LitModule


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model-based Asynchronous HPO')
    parser.add_argument('--data_path',
                        default='',
                        type=str,
                        help='path to dataset')
    parser.add_argument('--train_csv_file',
                        default='',
                        type=str,
                        help='path to train dataset')
    parser.add_argument('--test_csv_file',
                        default='',
                        type=str,
                        help='path to test dataset')
    parser.add_argument('--model_name',
                        metavar='MODEL',
                        default='resnet18',
                        help='model architecture: (default: resnet18)')
    parser.add_argument('--pretrained',
                        dest='pretrained',
                        action='store_true',
                        help='use pre-trained model')
    parser.add_argument('-j',
                        '--workers',
                        type=int,
                        default=4,
                        metavar='N',
                        help='how many training processes to use (default: 1)')
    parser.add_argument('--epochs',
                        default=90,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--image-size',
                        default=256,
                        type=int,
                        help='resolution of image')
    parser.add_argument('-b',
                        '--batch-size',
                        default=256,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 256) per gpu')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.1,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('--end-lr',
                        '--minimum learning-rate',
                        default=1e-8,
                        type=float,
                        metavar='END-LR',
                        help='initial learning rate')
    parser.add_argument(
        '--lr-schedule',
        default='step',
        type=str,
        metavar='SCHEDULE',
        choices=['step', 'linear', 'cosine', 'exponential'],
        help='Type of LR schedule: {}, {}, {} , {}'.format(
            'step', 'linear', 'cosine', 'exponential'),
    )
    parser.add_argument('--warmup',
                        default=0,
                        type=int,
                        metavar='E',
                        help='number of warmup epochs')
    parser.add_argument('--optimizer',
                        default='sgd',
                        type=str,
                        choices=('sgd', 'rmsprop', 'adamw'))
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')
    parser.add_argument('--wd',
                        '--weight-decay',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument(
        '--augmentation',
        type=str,
        default=None,
        choices=[None, 'autoaugment'],
        help='augmentation method',
    )
    parser.add_argument('--log_interval',
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--resume',
                        default=None,
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate',
                        dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--checkpoint-filename',
                        default='checkpoint.pth',
                        type=str)
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--checkpoint_dir',
                        default='work_dirs',
                        type=str,
                        help='output directory for model and log')
    args = parser.parse_args()
    return args


def train(train_csv_file,
          test_csv_file,
          val_fold=0.0,
          image_size=256,
          batch_size=64,
          num_workers=4,
          model_name='resnet50',
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
          checkpoint_dir=None,
          accumulate_grad_batches=1,
          auto_lr_find=False,
          auto_scale_batch_size=False,
          fast_dev_run=False,
          gpus=1,
          max_epochs=100,
          precision=16,
          stochastic_weight_avg=True,
          debug=True):

    pl.seed_everything(42)

    # ???????????????
    datamodule = LitDataModule(
        train_csv_file=train_csv_file,
        test_csv_file=test_csv_file,
        val_fold=val_fold,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    datamodule.setup()
    len_train_dl = len(datamodule.train_dataloader())

    # ????????????
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

    # ?????????ModelCheckpoint????????????????????????????????????
    #  monitor????????????????????????string?????????
    # ??????'val_loss'??????training_step() or validation_step()???????????????self.log('val_loss', loss)??????????????????
    # ?????????None????????????????????????epoch???????????????,
    model_checkpoint = ModelCheckpoint(checkpoint_dir,
                                       filename=f'{model_name}_{image_size}',
                                       monitor='val_loss',
                                       save_top_k=5)

    # ??????trainer
    trainer = pl.Trainer(
        accumulate_grad_batches=accumulate_grad_batches,  # ???k???batches??????????????????
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=auto_scale_batch_size,
        benchmark=True,
        callbacks=[model_checkpoint],  # ???????????????????????????????????????
        deterministic=True,
        fast_dev_run=fast_dev_run,
        gpus=gpus,  # ?????????gpu??????(int)???gpu????????????(list???str)
        max_epochs=2 if debug else max_epochs,  # ??????????????????
        precision=precision,
        stochastic_weight_avg=stochastic_weight_avg,
        limit_train_batches=0.1
        if debug else 1.0,  # ????????????/??????/??????/?????????????????????????????????????????????,??????????????????????????????
        limit_val_batches=0.1 if debug else 1.0,
    )

    # Trainer.tune()??????????????????????????????
    trainer.tune(module, datamodule=datamodule)

    # ????????????
    # Trainer.fit() ????????????
    # model->LightningModule??????;
    # train_dataloaders->?????????????????????
    # val_dataloaders->?????????????????????
    # ckpt_path->ckpt????????????(???????????????????????????)
    # datamodule->LightningDataModule??????
    trainer.fit(module, datamodule=datamodule)


if __name__ == '__main__':
    args = parse_args()
    train(train_csv_file=args.train_csv_file,
          test_csv_file=args.test_csv_file,
          model_name=args.model_name,
          image_size=args.image_size,
          batch_size=args.batch_size,
          pretrained=args.pretrained,
          checkpoint_dir=args.checkpoint_dir)
