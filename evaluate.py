import argparse
import os
import pathlib
import random

import numpy as np
import tqdm

import helpers
from flow_utils import *
from metrics.lpips.loss import PerceptualLoss


def main(opt):
    if opt.device:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.device:
        torch.cuda.manual_seed_all(opt.seed)
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    models = helpers.load_model_from_checkpoint(opt, device)
    print(opt)
    _, test_loader = helpers.get_loaders(opt, True)

    lpips_model = PerceptualLoss('lpips_weights', use_gpu=True)
    all_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    all_samples = {'psnr': [], 'ssim': [], 'lpips': []}
    gts = []
    for i, test_x in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        # test_x = next(testing_batch_generator)
        test_x = test_x.to(device)[:opt.n_eval]
        samples, metrics = helpers.eval_step(test_x, models, opt, device, lpips_model)
        for name in samples.keys():
            all_metrics[name] += metrics[name].cpu().detach().numpy().tolist()
            all_samples[name] += [
                (samples[name].cpu().detach().numpy() * 255).astype('uint8')]  # shape is different now, check this
        gts += [test_x.cpu().detach().numpy()]
    pathlib.Path(opt.log_dir).mkdir(exist_ok=True)
    to_save = np.concatenate(gts, axis=1)
    np.savez_compressed(os.path.join(opt.log_dir, 'gts.npz'), samples=to_save)

    for name, values in all_metrics.items():
        print(name, np.mean(values), '+/-', np.std(values) / np.sqrt(len(values)))
        to_save = np.concatenate(all_samples[name], axis=0)
        np.savez_compressed(os.path.join(opt.log_dir, f'{name}.npz'), samples=np.transpose(to_save, (1, 0, 4, 2, 3)))
        np.savez_compressed(os.path.join(opt.log_dir, f'results_{name}.npz'), values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=20, type=int, help='batch size')
    parser.add_argument('--data_root', default='mnist_data', help='root directory for data')
    parser.add_argument('--model_path', default='', help='path to model')
    parser.add_argument('--log_dir', default='', help='directory to save generations to')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=20, help='number of frames to predict')
    parser.add_argument('--data_threads', type=int, default=0, help='number of data loading threads')
    parser.add_argument('--nsample', type=int, default=100, help='number of samples')
    parser.add_argument('--device', action='store_true', help='if true, use gpu')
    opt = parser.parse_args()
    opt.n_eval = opt.n_past + opt.n_future
    main(opt)
