'''
Author: jianzhnie
Date: 2022-03-29 19:12:38
LastEditTime: 2022-03-29 19:17:34
LastEditors: jianzhnie
Description: 

'''

from torch.optim import lr_scheduler


def fetch_scheduler(optimizer, scheduler, T_max, T_0, min_lr):
    if scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=T_max,
                                                   eta_min=min_lr)
    elif scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=T_0,
                                                             eta_min=min_lr)
    elif scheduler == None:
        return None

    return scheduler
