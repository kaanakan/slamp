import os
import random

import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter

import args
import helpers
from flow_utils import *
from metrics.lpips.loss import PerceptualLoss


def main(opt):
    if opt.device:
        device = torch.device('cuda:0')
        print(device)
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

    models, optimizers = helpers.get_model(opt, device)
    # trainset, valset = helpers.get_datasets(opt)
    train_loader, val_loader = helpers.get_loaders(opt)

    def get_training_batch():
        while True:
            for sequence in train_loader:
                yield sequence.to(device)

    training_batch_generator = get_training_batch()

    writer = SummaryWriter('runs/' + opt.name)
    lpips_model = PerceptualLoss('lpips_weights', use_gpu=opt.device)
    # --------- training loop ------------------------------------
    best_psnr = -1
    best_ssim = -1
    best_lpips = 100
    t = tqdm.trange(opt.niter, desc='Bar desc', position=0, leave=True)
    total_iter = 0

    for epoch in t:
        for k, v in models.items():
            v.train()
        epoch_pixel_mse = 0
        epoch_flow_mse = 0
        epoch_mask_mse = 0
        epoch_kld = 0
        if opt.sch_sampling != 0:
            opt.sc_prob = opt.sch_sampling / (opt.sch_sampling + np.exp(epoch / opt.sch_sampling))

        for i in range(opt.epoch_size):
            total_iter += 1

            x = next(training_batch_generator)

            # train frame_predictor 
            pixel_mse, flow_mse, mask_mse, kld = helpers.train_step(x, models, optimizers, opt, device)

            epoch_pixel_mse += pixel_mse
            epoch_flow_mse += flow_mse
            epoch_mask_mse += mask_mse
            epoch_kld += kld
            writer.add_scalars('train/reconstruction', {
                'pixel': pixel_mse,
                'flow': flow_mse,
                'mask': mask_mse
            }, total_iter)
            writer.add_scalar('train/kld', kld, total_iter)

        t.set_description('pixel loss: %.5f | flow loss: %.5f | final loss: %.5f | kld loss: %.5f'
                          % (epoch_pixel_mse / opt.epoch_size, epoch_flow_mse / opt.epoch_size,
                             epoch_mask_mse / opt.epoch_size, epoch_kld / opt.epoch_size))

        if epoch % 10 == 0 and epoch != 0:
            for key, val in models.items():
                if any(model_type in key for model_type in ['prior', 'posterior', 'predictor']):
                    # if an lstm-variant
                    val.eval()

            eval_metrics, eval_samples = helpers.eval_model(val_loader, models, opt, device)

            writer.add_scalar('eval/psnr', eval_metrics['psnr'], epoch)
            writer.add_scalar('eval/ssim', eval_metrics['ssim'], epoch)
            writer.add_scalar('eval/lpips', eval_metrics['lpips'], epoch)

            to_save = models
            to_save['opt'] = opt
            to_save['epoch'] = epoch

            if eval_metrics['psnr'] > best_psnr:
                print('best psnr model, psnr=', eval_metrics['psnr'])
                torch.save(to_save,
                           '%s/best_psnr_model.pth' % (opt.log_dir))
                best_psnr = eval_metrics['psnr']
                np.savez_compressed(os.path.join(opt.log_dir, 'psnr_samples.npz'), samples=eval_samples['psnr'])

            if eval_metrics['ssim'] > best_ssim:
                print('best ssim model, ssim=', eval_metrics['ssim'])
                torch.save(to_save,
                           '%s/best_ssim_model.pth' % (opt.log_dir))
                best_ssim = eval_metrics['ssim']
                np.savez_compressed(os.path.join(opt.log_dir, 'ssim_samples.npz'), samples=eval_samples['ssim'])

            if eval_metrics['lpips'] < best_lpips:
                print('best lpips model, lpips=', eval_metrics['lpips'])

                torch.save(to_save,
                           '%s/best_lpips_model.pth' % (opt.log_dir))
                best_lpips = eval_metrics['lpips']
                np.savez_compressed(os.path.join(opt.log_dir, 'lpips_samples.npz'), samples=eval_samples['lpips'])

            torch.save(to_save,
                       '%s/model_%d.pth' % (opt.log_dir, epoch))


if __name__ == '__main__':
    opt = args.get_parser()
    print(opt)
    main(opt)
