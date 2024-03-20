import os
import re
import pdb
import logging
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class BasicDataset(Dataset):
    def __init__(self, root, scale=1, dim=3, direction='AtoB', type='train', norm='std', no_mask=False):
        self.norm = norm
        self.scale = scale
        self.dim = dim
        self.direction = direction
        self.type = type
        self.no_mask = no_mask

        dirs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in dirs]

        logging.info(f'Creating dataset with {len(self.imgs)} examples')

    def __len__(self):
        return len(self.imgs)

    @classmethod
    def preprocess_C1(cls, pil_img, norm):
        H, W = pil_img.shape
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        temp1 = np.zeros((3,H,W))
        img_trans = img_nd.transpose((2, 0, 1)).astype(np.float32)

        if norm == 'std':
            img_trans = (img_trans - np.mean(img_trans)) / np.std(img_trans)
        else:
            img_trans = img_trans / 255

        temp1[0, :, :] = img_trans
        temp1[1, :, :] = img_trans
        temp1[2, :, :] = img_trans
        img_trans = temp1

        return img_trans

    @classmethod
    def preprocess_C3(cls, pil_img, norm):

        H, W, _ = pil_img.shape
        img_nd = np.array(pil_img)

        img_trans = img_nd.transpose((2, 0, 1)).astype(np.float32)
        temp1 = np.zeros_like(img_trans)
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

        temp1[0, :, :] = img_trans_1
        temp1[1, :, :] = img_trans_2
        temp1[2, :, :] = img_trans_3

        return temp1

    def __getitem__(self, index):
        img_path = self.imgs[index]
        name = img_path.split('/')[-1]
        temp = Image.open(img_path)
        temp = np.asarray(temp)
        if self.no_mask:
            image = np.float32(temp)
            mask = torch.IntTensor([0])
        elif len(temp.shape) > 2:
            h, w, _ = temp.shape
            w_m = int(w/2)
            if self.direction == 'AtoB':
                image = np.float32(temp[:, 0:w_m, :])
                mask = np.float32(temp[:, w_m:w, 0])
            else:
                mask = np.float32(temp[:, 0:w_m, 0])
                image = np.float32(temp[:, w_m:w, :])
        else:
            h, w = temp.shape
            w_m = int(w/2)
            if self.direction == 'AtoB':
                image = np.float32(temp[:, 0:w_m])
                mask = np.float32(temp[:, w_m:w])
            else:
                mask = np.float32(temp[:, 0:w_m])
                image = np.float32(temp[:, w_m:w])
        if self.type == 'test':
            delta = int(re.split(r'[_.]', img_path)[-3])
            return {'name': name, 'image': image, 'mask': torch.from_numpy(mask), 'delta': torch.IntTensor([delta])}
        else:
            if self.dim == 3:
                image = self.preprocess_C3(image, self.norm)
            elif self.dim == 1:
                image = self.preprocess_C1(image, self.norm)
            return {'image': torch.from_numpy(image), 'mask': torch.from_numpy(mask)}


if __name__ == '__main__':
    test_dir = 'data/Revise/test'
    test = BasicDataset(test_dir, scale=1, dim=1, direction='AtoB', type='test', no_mask=False)
    test_loader = DataLoader(test, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
    for batch in test_loader:
        pdb.set_trace()