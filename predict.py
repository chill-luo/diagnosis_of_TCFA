import os
import cv2
import argparse
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import morphology

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.UNet_3 import UNet_3
from models.UNet_5 import UNet
from models.UNetPlus import NestedUNet
from models.simple_unet import simple_unet
from dataset import BasicDataset
from utils import mask_vis, Timer, XMLParser
from metrics import sub_iou, iou_for_lip

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='UNet', choices=['UNet', 'simple_unet', 'UNet_3', 'NestedUNet'],
                        help='Which network to train')
    parser.add_argument('--input', '-i', metavar='INPUT', default='ER/',type=str,
                        help='types of input images')
    parser.add_argument('--load_pth_fib', type=str,default='trained_on_ER_594_SNR_8/',
                        help='model weights directory')
    parser.add_argument('--load_pth_liq', type=str,default='trained_on_ER_594_SNR_8/',
                        help='model weights directory')
    parser.add_argument('--save_pth', type=str,default='trained_on_ER_594_SNR_8/',
                        help='Where to save results')

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which GPU to be used. Such as -gpu 0')
    parser.add_argument('--direction', type=str, default='AtoB',
                        help='AtoB or BtoA')
    parser.add_argument('--epoch', type=int, default=0, help='which model to choose')
    parser.add_argument('--n_class', type=int, default=2, help='number of class')
    parser.add_argument('--channel', type=int, default=1, help='Channel size')
    parser.add_argument('--norm', type=str, default='std', help='which normalization...')
    parser.add_argument('--only_segment', action='store_true', help='Whether to analyse the segmentation result.')
    parser.add_argument('--need_score', action='store_true', help='Whether to output score maps.')

    return parser.parse_args()


def preprocess(imgs, channel=1, norm='std'):
    imgs = np.array(imgs)
    if channel == 1:
        N, H, W = imgs.shape
        for n in range(N):
            img_nd = imgs[n]
            if len(img_nd.shape) == 2:
                img_nd = np.expand_dims(img_nd, axis=2)
            # HWC to CHW
            temp = np.zeros((3, H, W))
            img_trans = img_nd.transpose((2, 0, 1)).astype(np.float32)
            if norm == 'std':
                img_trans = (img_trans - np.mean(img_trans)) / np.std(img_trans)
            else:
                img_trans = img_trans / 255
            temp[0, :, :] = img_trans
            temp[1, :, :] = img_trans
            temp[2, :, :] = img_trans
            temp = torch.from_numpy(temp).unsqueeze(0)
            if n == 0:
                tensors = temp
            else:
                tensors = torch.cat((tensors, temp), dim=0)
    elif channel == 3:
        N, H, W, _ = imgs.shape
        for n in range(N):
            img_nd = imgs[n]
            # HWC to CHW
            img_trans = img_nd.transpose((2, 0, 1)).astype(np.float32)
            temp = np.zeros_like(img_trans)
            if norm == 'std':
                img_trans_1 = img_trans[0, :, :]
                img_trans_1 = (img_trans_1 - np.mean(img_trans_1)) / np.std(img_trans_1)
                img_trans_2 = img_trans[1, :, :]
                img_trans_2 = (img_trans_2 - np.mean(img_trans_2)) / np.std(img_trans_2)
                img_trans_3 = img_trans[2, :, :]
                img_trans_3 = (img_trans_3 - np.mean(img_trans_3)) / np.std(img_trans_3)
            else:
                img_trans_1 = img_trans / 255
                img_trans_2 = img_trans / 255
                img_trans_3 = img_trans / 255
            temp[0, :, :] = img_trans_1
            temp[1, :, :] = img_trans_2
            temp[2, :, :] = img_trans_3
            temp = torch.from_numpy(temp).unsqueeze(0)
            if n == 0:
                tensors = temp
            else:
                tensors = torch.cat((tensors, temp), dim=0)
    return tensors


def cut_image(images, masks, resolutions, channel=1):
    images = images.clone()
    n, h, w = images.shape[:3]
    for k in range(n):
        length = int(1.0 * resolutions[k])
        mask_tmp = np.array(masks[k, 0, :, :].cpu()).astype(np.bool8)
        mask_tmp = morphology.remove_small_objects(mask_tmp, min_size=1000)
        # pdb.set_trace()
        for i in range(h):
            flag = np.where(mask_tmp[i, :]>0)[0]
            if len(flag) == 0:
                continue
            else:
                start = flag.min()
                end = min(start+length, w-1)
                if channel == 1:
                    images[k, i, end:] = 0
                elif channel == 3:
                    images[k, i, end:, :] = 0
    return images


if __name__ == "__main__":
    args = get_args()
    print('GPU id:', args.gpu)
    torch.cuda.set_device(args.gpu)
    model_name = args.model

    # load dateset
    test_dir = os.path.join('data', args.input)
    test = BasicDataset(test_dir, scale=1, dim=args.channel, direction=args.direction, type='test', no_mask=args.only_segment)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # load model networks
    model_f = eval(model_name)(output_c=2)
    model_l = eval(model_name)(output_c=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_f = model_f.to(device=device)
    model_l = model_l.to(device=device)

    if args.epoch == 0:
        load_path_f = os.path.join('checkpoints', model_name, args.load_pth_fib, 'model_best.pth')
        load_path_l = os.path.join('checkpoints', model_name, args.load_pth_liq, 'model_best.pth')
    else:
        load_path_f = os.path.join('checkpoints', model_name, args.load_pth_fib, 'CP_epoch' + str(args.epoch) + '.pth')
        load_path_l = os.path.join('checkpoints', model_name, args.load_pth_liq, 'CP_epoch' + str(args.epoch) + '.pth')

    m_f = nn.DataParallel(model_f)
    m_f.load_state_dict(torch.load(load_path_f, map_location=device), strict=False)
    model_f = m_f.module
    m_l = nn.DataParallel(model_l)
    m_l.load_state_dict(torch.load(load_path_l, map_location=device), strict=False)
    model_l = m_l.module
    logging.info('GPU id is: DataParallel')

    logging.info("Model loaded !")
    print('Model_name:', model_name)
    print('Model_path:', load_path_f, ' and ', load_path_l)

    input_path = os.path.join('data', args.input)
    in_files = os.listdir(input_path)
    print('img number:', len(in_files))
    print('img file:', input_path)

    # save path
    save_path = os.path.join('results', model_name, args.save_pth)
    os.makedirs(save_path, exist_ok=True)
    print('saving to:', save_path)
    iou_all = 0
    log_file = XMLParser()
    log_file.add_node('results')
    log_file.add_node('final')

    timer = Timer()
    model_f.eval()
    model_l.eval()
    with tqdm(total=len(in_files), desc='Predicting...') as pbar:
        for batch in test_loader:
            names, imgs_ori, true_masks, deltas = batch['name'], batch['image'], batch['mask'], batch['delta']
            imgs = preprocess(imgs_ori, channel=args.channel, norm=args.norm)
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            deltas = deltas.to(device=device, dtype=torch.int32)

            with torch.no_grad():
                output_fs = model_f(imgs)     # N*C*H*W
                scoremap_fs = torch.softmax(output_fs, dim=1)[:, 1, :, :].cpu()
                output_fs = torch.argmax(output_fs, dim=1, keepdim=True).type(torch.float32)     # N*1*H*W
                
                cut_imgs_ori = cut_image(imgs_ori, output_fs, deltas)
                cut_imgs = preprocess(cut_imgs_ori, channel=args.channel, norm=args.norm)
                cut_imgs = cut_imgs.to(device=device, dtype=torch.float32)
                
                output_ls = model_l(cut_imgs)
                scoremap_ls = torch.softmax(output_ls, dim=1)[:, 1, :, :].cpu()
                output_ls = torch.argmax(output_ls, dim=1, keepdim=True).type(torch.float32)

                outputs = torch.zeros(output_fs.shape).to(device=device, dtype=torch.float32)     # N*1*H*W
                outputs[output_ls==1] = 2
                outputs[output_fs==1] = 1

                for i in range(imgs.shape[0]):
                    name = names[i]
                    img_save = Image.fromarray(np.float32(imgs_ori)[i]).convert('RGB')
                    img_save.save(os.path.join(save_path, name[0:-4] + '_img' + name[-4:]))
                    cut_img_save = Image.fromarray(np.float32(cut_imgs_ori)[i]).convert('RGB')
                    cut_img_save.save(os.path.join(save_path, name[0:-4] + '_img_cut' + name[-4:]))
                    
                    out = outputs[i]
                    out_org = out.squeeze().cpu().numpy()
                    out_vis = mask_vis(out[0])
                    out_vis  = out_vis.squeeze().cpu().numpy()
                    out_vis = out_vis.transpose((1, 2, 0))
                    cv2.imwrite(os.path.join(save_path, name[0:-4] + '_pred' + name[-4:]), np.uint8(out_vis))
                    cv2.imwrite(os.path.join(save_path, name[0:-4] + '_mask' + name[-4:]), np.uint16(out_org))

                    if args.need_score:
                        np.save(os.path.join(save_path, name[0:-4] + '_fmap.npy'), np.array(scoremap_fs[i]))
                        np.save(os.path.join(save_path, name[0:-4] + '_lmap.npy'), np.array(scoremap_ls[i]))
                    if args.only_segment:
                        continue

                    label = true_masks[i]
                    iou = torch.zeros(3)
                    iou[1] = sub_iou(1, outputs[i].unsqueeze(0), label.unsqueeze(0), argmax=False)
                    iou[2] = iou_for_lip(outputs[i].unsqueeze(0), label.unsqueeze(0), argmax=False)
                    miou = iou[1:].mean()
                    iou[0] = 0.7 * iou[1] + 0.3 * iou[2]
                    if abs(iou[0] - miou) <= 1:
                        if isinstance(iou_all, int):
                            iou_all = iou
                        else:
                            iou_all = np.vstack((iou_all, iou))

                    node = 'results/image:' + name
                    log_file.add_node(node)
                    if abs(iou[0] - miou) > 1:
                        log_file.add_node(node + '/error', content='No target detected.')
                    else:
                        log_file.add_node(node + '/miou', content=iou[0].item())
                        log_file.add_node(node + '/iou_1', content=iou[1].item())
                        log_file.add_node(node + '/iou_2', content=iou[2].item())

                    mask = mask_vis(label)
                    mask = mask.squeeze().cpu().numpy()
                    mask = mask.transpose((1, 2, 0))
                    cv2.imwrite(os.path.join(save_path, name[0:-4] + '_gt' + name[-4:]), np.uint8(mask))
            pbar.update(imgs.shape[0])
    print('Use time: ', timer.measure())

    if not args.only_segment:    
        iou_avg = iou_all.mean(0)
        logging.info(f'mIoU: {iou_avg[0]}')
        for ind in range(1, args.n_class):
            logging.info(f'IoU_{ind}: {iou_avg[ind]}')

        log_file.add_node('final/miou', content=iou_avg[0])
        for ind in range(1, args.n_class):
            log_file.add_node(f'final/iou{ind}', content=iou_avg[ind])
        log_file.save(os.path.join(save_path, 'results.xml'))