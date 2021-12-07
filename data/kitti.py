import os
import numpy as np
from PIL import Image
import glob
import itertools
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Kittiloader(Dataset):
    def __init__(self, data_path, seq_len, train, w=310, h=92, ext='jpg', aug=True):
        self.data_path = data_path
        self.seq_len = seq_len
        self.train = train
        self.h = h
        self.w = w
        self.ext = ext
        self.to_tensor = to_tensor = transforms.ToTensor()
        self.aug = aug
        train_folders = ['2011_09_26_drive_0001_sync', '2011_09_26_drive_0005_sync',
                         '2011_09_26_drive_0011_sync', '2011_09_26_drive_0014_sync',
                         '2011_09_26_drive_0015_sync', '2011_09_26_drive_0017_sync',
                         '2011_09_26_drive_0018_sync', '2011_09_26_drive_0019_sync',
                         '2011_09_26_drive_0022_sync', '2011_09_26_drive_0028_sync',
                         '2011_09_26_drive_0032_sync', '2011_09_26_drive_0035_sync',
                         '2011_09_26_drive_0039_sync', '2011_09_26_drive_0051_sync',
                         '2011_09_26_drive_0057_sync', '2011_09_26_drive_0061_sync',
                         '2011_09_26_drive_0070_sync', '2011_09_26_drive_0079_sync',
                         '2011_09_26_drive_0087_sync', '2011_09_26_drive_0091_sync',
                         '2011_09_26_drive_0095_sync', '2011_09_26_drive_0104_sync',
                         '2011_09_26_drive_0113_sync', '2011_09_28_drive_0001_sync',
                         '2011_09_29_drive_0004_sync', '2011_09_29_drive_0026_sync',
                         '2011_09_30_drive_0020_sync', '2011_09_30_drive_0028_sync',
                         '2011_09_30_drive_0033_sync', '2011_09_30_drive_0034_sync',
                         '2011_10_03_drive_0034_sync', '2011_10_03_drive_0042_sync']
        val_folders = train_folders
        test_folders = ['2011_09_26_drive_0002_sync', '2011_09_26_drive_0009_sync',
                        '2011_09_26_drive_0013_sync', '2011_09_26_drive_0020_sync',
                        '2011_09_26_drive_0023_sync', '2011_09_26_drive_0027_sync',
                        '2011_09_26_drive_0029_sync', '2011_09_26_drive_0036_sync',
                        '2011_09_26_drive_0046_sync', '2011_09_26_drive_0048_sync',
                        '2011_09_26_drive_0052_sync', '2011_09_26_drive_0056_sync',
                        '2011_09_26_drive_0059_sync', '2011_09_26_drive_0064_sync',
                        '2011_09_26_drive_0084_sync', '2011_09_26_drive_0086_sync',
                        '2011_09_26_drive_0093_sync', '2011_09_26_drive_0096_sync',
                        '2011_09_26_drive_0101_sync', '2011_09_26_drive_0106_sync',
                        '2011_09_26_drive_0117_sync', '2011_09_28_drive_0002_sync',
                        '2011_09_29_drive_0071_sync', '2011_09_30_drive_0016_sync',
                        '2011_09_30_drive_0018_sync', '2011_09_30_drive_0027_sync',
                        '2011_10_03_drive_0027_sync', '2011_10_03_drive_0047_sync']
        self.folders = {'train':train_folders, 'val':val_folders, 'test':test_folders}
        self.get_data_paths()
        if train in ['test', 'val']:
            self.arange_val_paths()


    def arange_val_paths(self):
        dd = {}
        for path in self.data:
            p, im = path[:-15], path[-14:]
            if p not in dd:
                dd[p] = p + im
        self.data = list(dd.values())



    def get_data_paths(self):
        with open(os.path.join(os.path.dirname(__file__), 'lengths.txt')) as f:
            content = f.readlines()
            lengths = [x.strip() for x in content]

        with open(os.path.join(os.path.dirname(__file__), 'static-ranges.txt')) as f:
            content = f.readlines()
        ranges = [x.strip() for x in content]
        ranges_dict = {}

        for line in ranges:
            day, folder, start, _, end = line.split()
            if day not in ranges_dict:
                ranges_dict[day] = {}
            if folder not in ranges_dict[day]:
                ranges_dict[day][folder] = []
            start, end = int(start), int(end)
            ranges_dict[day][folder].append((start, end))

        data = []
        for line in lengths:
            path, l = line.split()
            day, folder, _, _ = path.split('/')
            if folder not in self.folders[self.train]:
                continue
            if folder not in ranges_dict[day]:
                for i in range(int(l)-self.seq_len):
                    data.append(os.path.join(path, '{:010d}.{}'.format(i, self.ext)))
                    data.append(os.path.join(path.replace('image_02', 'image_03'), '{:010d}.{}'.format(i, self.ext)))
            else:
                begin = 0
                for rng in ranges_dict[day][folder]:
                    start, end = rng
                    for i in range(begin, start-self.seq_len):
                        data.append(os.path.join(path, '{:010d}.{}'.format(i, self.ext)))
                        data.append(os.path.join(path.replace('image_02', 'image_03'), '{:010d}.{}'.format(i, self.ext)))
                    begin = end+1
                for i in range(begin, int(l)-self.seq_len):
                    data.append(os.path.join(path, '{:010d}.{}'.format(i, self.ext)))
                    data.append(os.path.join(path.replace('image_02', 'image_03'), '{:010d}.{}'.format(i, self.ext)))

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
        p, ext = fullpath.split('.')
        t0 = int(p[-10:])
        folder = p[:-10]
        for t in range(self.seq_len):
            img_path = os.path.join(self.data_path, folder, '{:010d}.{}'.format(t0+t, ext))
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
