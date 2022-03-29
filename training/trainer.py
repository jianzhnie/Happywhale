'''
Author: jianzhnie
Date: 2022-03-29 18:49:10
LastEditTime: 2022-03-29 19:08:04
LastEditors: jianzhnie
Description:

'''
import copy
import gc
import time
from collections import defaultdict

import numpy as np
import torch
import tqdm


def run_training(model, optimizer, criterion, scheduler, train_loader,
                 valid_loader, n_accumulate, num_epochs, device):
    if torch.cuda.is_available():
        print('[INFO] Using GPU: {}\n'.format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(model,
                                           optimizer,
                                           criterion,
                                           scheduler,
                                           dataloader=train_loader,
                                           n_accumulate=n_accumulate,
                                           epoch=epoch,
                                           device=device)

        val_epoch_loss = valid_one_epoch(model,
                                         optimizer,
                                         criterion,
                                         valid_loader,
                                         epoch=epoch,
                                         device=device)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)

        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(
                f'{epoch} Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})'
            )
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = 'Loss{:.4f}_epoch{:.0f}.pth'.format(best_epoch_loss, epoch)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print('Model Saved')

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60,
        (time_elapsed % 3600) % 60))
    print('Best Loss: {:.4f}'.format(best_epoch_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def train_one_epoch(model, optimizer, criterion, scheduler, dataloader,
                    n_accumulate, epoch, device):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)

        batch_size = images.size(0)

        outputs = model(images, labels)
        loss = criterion(outputs, labels)
        loss = loss / n_accumulate

        loss.backward()

        if (step + 1) % n_accumulate == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch,
                        Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])

    return epoch_loss


@torch.inference_mode()
def valid_one_epoch(model, optimizer, criterion, dataloader, epoch, device):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)

        batch_size = images.size(0)

        outputs = model(images, labels)
        loss = criterion(outputs, labels)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch,
                        Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])

    return epoch_loss
