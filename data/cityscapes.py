import os
import numpy as np
from PIL import Image
import glob
import itertools
import torch
from torch.utils.data import Dataset
from torchvision import transforms



class CityScapesLoader(Dataset):
    def __init__(self, data_path, seq_len, train, w=256, h=128, ext='jpg', aug=True):
        assert seq_len <= 30
        self.data_path = data_path
        self.seq_len = seq_len
        self.train = train
        self.h = h
        self.w = w
        self.ext = ext
        self.to_tensor = to_tensor = transforms.ToTensor()
        self.aug = aug
        train_folders = ['train']
        test_folders = ['test']
        
        self.folders = {'train':train_folders, 'test':test_folders}
        self.get_data_paths(train)
        
    
    def get_data_paths(self, train):
        data = []
        stride = 30 - self.seq_len + 1
        if train == 'train':
            dirs = []
            for d in self.folders['train']:
                dirs += sorted(glob.glob(os.path.join(self.data_path, d, '*')))
            for d in dirs:
                imgs = sorted(glob.glob(os.path.join(d, '*')))
                total_len = len(imgs)
                data += [imgs[i:i+stride] for i in range(0, total_len, 30)]
            data = [item for sublist in data for item in sublist]
        else:
            dirs = []
            for d in self.folders['test']:
                dirs += sorted(glob.glob(os.path.join(self.data_path, d, '*')))
            for d in dirs:
                imgs = sorted(glob.glob(os.path.join(d, '*')))
                total_len = 20 * 30 #len(imgs)
                data += [imgs[i] for i in range(0, total_len, 30)]
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        color_jitter = transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                                          saturation=(0.8, 1.2), hue=(-0.1, 0.1))
        transform = transforms.ColorJitter.get_params(color_jitter.brightness, color_jitter.contrast,
                                              color_jitter.saturation, color_jitter.hue)
        do_aug = self.aug and self.train =='train' and np.random.random() > 0.5
        do_flip = self.aug and self.train =='train' and np.random.random() > 0.5
        x = torch.zeros((self.seq_len,3, self.h, self.w))
        fullpath = self.data[index]
        p = fullpath.split('/')
        folder, p = os.path.join('/',*p[:-1]), p[-1]
        p, ext = p.split('.')
        im_name = p.split('_')
        t0 = int(im_name[2])
        im_name[2] = '{:06d}'
        im_name[-1] = im_name[-1] + '.{}'.format(ext)
        im_name = '_'.join(im_name)
        for t in range(self.seq_len):
            img_path = os.path.join(folder, im_name.format(t0 + t))
            img = Image.open(img_path).convert('RGB').resize((self.w, self.h))
            if do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self.to_tensor(img)
            if do_aug:
                img = transform(img)
            x[t, :, :, :] = img
        return x



def collate_fn(videos):
    """
    Collate function for the PyTorch data loader.
    Merges all batch videos in a tensor with shape (length, batch, channels, width, height) and converts their pixel
    values to [0, 1].
    Parameters
    ----------
    videos : list
        List of uint8 NumPy arrays representing videos with shape (length, batch, width, height, channels).
    Returns
    -------
    torch.*.Tensor
        Batch of videos with shape (length, batch, channels, width, height) and float values lying in [0, 1].
    """
    
    seq_len = len(videos[0])
    batch_size = len(videos)
    nc = 1 if videos[0].ndim == 3 else 3
    w = videos[0].shape[3]
    h = videos[0].shape[2]
    tensor = torch.zeros((seq_len, batch_size, nc, h, w))
    for i, video in enumerate(videos):
        if nc == 1:
            tensor[:, i, 0] += video
        if nc == 3:
            tensor[:, i] += video
    return tensor
