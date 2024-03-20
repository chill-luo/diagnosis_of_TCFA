import os
import sys
import argparse
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.UNet_3 import UNet_3
from models.UNet_5 import UNet
from models.UNetPlus import NestedUNet
from models.simple_unet import simple_unet
from dataset import BasicDataset
from metrics import miou_sp
from utils import  mask_vis, Timer

root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "models"))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='UNet',
                        choices=['UNet', 'simple_unet', 'UNet_3', 'NestedUNet'], help='which network to train')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=250,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--lr', metavar='LR', type=float, nargs='?', default=1e-2,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--save_pth', type=str, default='weigths/',
                        help='save weights to...')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-g','--gpu', type=str, default=0,
                        help='which GPU to be used. Sush as -gpu 0')
    parser.add_argument('--train_dir',  default='data/Seg_train/', nargs='+', type=str,
                        help='filenames of train images', required=False)
    parser.add_argument('--val_dir', default='data/Seg_test/', nargs='+', type=str,
                        help='filenames of val images', required=False)
    parser.add_argument('--channel', type=int, default=1, help='Channel size')
    parser.add_argument('--n_class', type=int, default=2, help='number of class')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--norm', type=str, default='std', help='which normalization...')

    return parser.parse_args()


def random_transform(images, masks):
    n, h, w = masks.shape
    images_tmp = torch.zeros_like(images).to(images.device)
    masks_tmp = torch.zeros_like(masks).to(masks.device)
    for i in range(n):
        shift = np.random.randint(1, h)
        images_tmp[i, :, :h-shift, :] = images[i, :, shift:, :]
        images_tmp[i, :, h-shift:, :] = images[i, :, :shift, :]
        masks_tmp[i, :h-shift, :] = masks[i, shift:, :]
        masks_tmp[i, h-shift:, :] = masks[i, :shift, :]
    return images_tmp, masks_tmp


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            mious = miou_sp(mask_pred, true_masks)
            miou_fix = 0.7 * mious[0] + 0.3 * mious[1] if len(mious) == 2 else mious.mean()
            tot += miou_fix

            pbar.update()

    return tot / n_val


def train(model,
          device,
          epochs,
          batch_size,
          lr,
          direction,
          channel
        ):
    train = BasicDataset(train_dir, scale=1, dim=channel, direction=direction)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8,drop_last=True, pin_memory=True)
    val = BasicDataset(val_dir, scale=1, dim=channel, direction=direction,type='val')
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    n_train = len(train)
    n_val = len(val)

    temp = train_dir
    writer = SummaryWriter(comment=f'_{model_name}_LR_{lr}_input_{temp}_save2_{dir_checkpoint}')
    global_step = 0
    logging.info(f'''Starting training:
            Data preprocess  {args.norm}
            Model name:      {model_name}
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Device:          {device.type}
            training dir:    {train_dir}
            val dir:    {val_dir}
            saving weights to:  {dir_checkpoint}
            GPU id:          {args.gpu}
            direction:       {direction}
            channel:         {channel}
        ''')
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if args.n_class == 3:
        weight_list = [1, 50, 50]
    elif args.n_class == 4:
        weight_list = [1, 50, 50, 100]
    elif args.n_class == 2:
        weight_list = [1, 50]
    W = torch.tensor(weight_list).to(device=device, dtype=torch.float32)

    logging.info(f'Loss_function: CrossEntropyLoss')
    criterion = nn.CrossEntropyLoss(weight = W)
    prev_score = float('-inf')

    timer = Timer()
    best_epoch = 0
    best_iou = 0.0
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                global_step += 1
                imgs = batch['image']
                true_masks = batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                imgs, true_masks = random_transform(imgs, true_masks)

                masks_pred = model(imgs)
                if args.n_class == -1:
                    masks_hard = torch.argmax(masks_pred, dim=1).type(torch.float32)
                    loss1 = criterion(masks_pred, true_masks)
                    thicks_p = (masks_hard == 1).sum(2)
                    thicks_g = (true_masks == 1).sum(2)
                    loss2 = abs(thicks_g - thicks_p).type(torch.float32).mean()
                    loss = loss1 * 0.5 + loss2 * 0.5
                else:
                    loss = criterion(masks_pred, true_masks)

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])

        # validation
        if (epoch+1) % 1 == 0:

            val_score = eval_net(model, val_loader, device)

            scheduler.step()

            iter_val = iter(val_loader)
            batch = next(iter_val)
            imgs = batch['image']
            label = batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            val_pred = model(imgs)
            llr = optimizer.param_groups[0]['lr']

            logging.info(f'Model:{model_name} // learning rate: {llr}')
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            logging.info('Validation mIoU: {:.5f}'.format(val_score))
            writer.add_scalar('mIoU/valid', val_score, global_step)
            writer.add_images('images', imgs, global_step)

            label = label[0,:,:]
            label = mask_vis(label)
            label = label.unsqueeze(0)

            masks_pred = torch.argmax(val_pred, dim=1, keepdim=True).type(torch.float32)
            masks_pred = masks_pred.squeeze(0)[0,:,:]
            masks_pred = mask_vis(masks_pred)
            masks_pred = masks_pred.unsqueeze(0)
            image = torch.cat((imgs, label, masks_pred), 0)
            image = utils.make_grid(image, nrow=image.shape[0], padding=2)
            utils.save_image(image, os.path.join(dir_checkpoint, 'mid', str(epoch+1) + '.png'))
        
        # save the best model weight
        is_better = val_score > prev_score
        if is_better:
            prev_score = val_score
            torch.save(model.state_dict(), os.path.join(dir_checkpoint, 'model_best.pth'))
            logging.info(f'Best model saved !')
            best_epoch = epoch + 1
            best_iou = val_score

        # save model weight per 30 epoch
        if (epoch + 1) % 30 == 0:
            torch.save(model.state_dict(),
                       os.path.join(dir_checkpoint, f'CP_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')
        logging.info(f'Using time: {timer.measure()}/{timer.measure((epoch+1) / args.epochs)}')
    logging.info(f'Best model was epoch {best_epoch}, the score is {best_iou}.')
    writer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    model_name = args.model
    name = args.train_dir[0]

    train_dir = 'data/' + name  # input data directory
    name = args.val_dir[0]
    val_dir = 'data/' + name
    dir_checkpoint = os.path.join('checkpoints', model_name, args.save_pth)  # save weights to...
    os.makedirs(dir_checkpoint, exist_ok=True)
    os.makedirs(os.path.join(dir_checkpoint, 'mid'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = eval(model_name)(output_c=args.n_class)

    if len(args.gpu) > 1:
        ids = list(args.gpu)
        ids = [ids[0], ids[2]]
        id = [int(x) for x in ids]
        device = torch.device(f'cuda:{id[0]}' if torch.cuda.is_available() else 'cpu')
        model = nn.DataParallel(model, device_ids=id)
        model = model.to(device)
        print('GPU id:', args.gpu)
    else:
        id = int(args.gpu)
        torch.cuda.set_device(id)
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print('GPU id:', args.gpu)

    train(model=model,
          epochs=args.epochs,
          batch_size=args.batchsize,
          lr=args.lr,
          device=device,
          direction=args.direction, 
          channel=args.channel)
