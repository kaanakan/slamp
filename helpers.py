import random

import numpy as np
from torch.utils.data import DataLoader

import data.base as data
import metric_helpers
import utils
from flow_utils import *


def get_datasets(opt, is_test=False):
    if opt.dataset in ['kitti', 'cityscapes']:
        return data.load_dataset(opt, 'train'), data.load_dataset(opt, 'test' if is_test else 'val')
    train_dataset = data.load_dataset(opt, 'train')
    trainset = train_dataset.get_fold('train')
    val_dataset = data.load_dataset(opt, 'test' if is_test else 'val')
    valset = val_dataset.get_fold('test' if is_test else 'val')
    return trainset, valset


def get_loaders(opt, is_test=False):
    trainset, valset = get_datasets(opt, is_test)
    if is_test:
        assert len(valset) % opt.batch_size == 0, f"Batch size should be a divisor of number of samples. " \
                                                  f"num_samples:{len(valset)}, batch_size:{opt.batch_size}"
    if opt.is_real_dataset:
        import data.kitti as kitti_dataset
        collate_fn = kitti_dataset.collate_fn
    else:
        collate_fn = data.collate_fn
    train_loader = DataLoader(trainset, batch_size=opt.batch_size, collate_fn=collate_fn,
                              num_workers=opt.data_threads, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=opt.batch_size, collate_fn=collate_fn,
                            num_workers=opt.data_threads, shuffle=False, drop_last=True if is_test else False,
                            pin_memory=True)
    return train_loader, val_loader


def get_model(opt, device):
    if opt.dataset in ['kitti', 'cityscapes']:
        return get_model_real(opt, device)
    else:
        return get_model_generic(opt, device)


def get_model_generic(opt, device):
    import models.lstm as lstm_models
    import models.mask_predictor as mask_predictor_model
    if opt.model == 'dcgan':
        import models.dcgan_64 as model
    else:
        import models.vgg_64 as model
    # lstms
    models = {}
    static_frame_predictor = lstm_models.lstm(opt.g_dim_app + opt.z_dim_app, opt.g_dim_app, opt.rnn_size,
                                              opt.predictor_rnn_layers, opt.batch_size, device)

    dynamic_frame_predictor = lstm_models.lstm(opt.g_dim_motion + opt.z_dim_motion, opt.g_dim_motion, opt.rnn_size,
                                               opt.predictor_rnn_layers, opt.batch_size, device)

    static_frame_predictor.apply(utils.init_weights)
    dynamic_frame_predictor.apply(utils.init_weights)

    posterior_app = lstm_models.gaussian_lstm(opt.g_dim_app, opt.z_dim_app, opt.rnn_size,
                                              opt.posterior_rnn_layers, opt.batch_size, device)
    prior_app = lstm_models.gaussian_lstm(opt.g_dim_app, opt.z_dim_app, opt.rnn_size,
                                          opt.prior_rnn_layers, opt.batch_size, device)

    posterior_motion = lstm_models.gaussian_lstm(opt.g_dim_motion, opt.z_dim_motion, opt.rnn_size,
                                                 opt.posterior_rnn_layers, opt.batch_size, device)
    prior_motion = lstm_models.gaussian_lstm(opt.g_dim_motion, opt.z_dim_motion, opt.rnn_size,
                                             opt.prior_rnn_layers, opt.batch_size, device)

    posterior_app.apply(utils.init_weights)
    prior_app.apply(utils.init_weights)

    posterior_motion.apply(utils.init_weights)
    prior_motion.apply(utils.init_weights)
    # convs
    pixel_encoder = model.encoder(opt.g_dim_app, opt.channels)
    motion_encoder = model.encoder(opt.g_dim_motion, opt.channels * 2)

    pixel_decoder = model.decoder(opt.g_dim_app, opt.channels)  # 3 channels for RGB, 1 channel for mnist
    flow_decoder = model.decoder(opt.g_dim_motion, 2, act=None)  # 2 channel flow, x and y
    mask_decoder = mask_predictor_model.mask_predictor(opt.channels * 2)

    pixel_encoder.apply(utils.init_weights)
    motion_encoder.apply(utils.init_weights)

    pixel_decoder.apply(utils.init_weights)
    flow_decoder.apply(utils.init_weights)
    mask_decoder.apply(utils.init_weights)

    models['static_frame_predictor'] = static_frame_predictor.to(device)
    models['dynamic_frame_predictor'] = dynamic_frame_predictor.to(device)
    models['posterior_app'] = posterior_app.to(device)
    models['prior_app'] = prior_app.to(device)
    models['posterior_motion'] = posterior_motion.to(device)
    models['prior_motion'] = prior_motion.to(device)

    models['pixel_encoder'] = pixel_encoder.to(device)
    models['motion_encoder'] = motion_encoder.to(device)

    models['pixel_decoder'] = pixel_decoder.to(device)
    models['flow_decoder'] = flow_decoder.to(device)
    models['mask_decoder'] = mask_decoder.to(device)

    optims = get_optimizers(models, opt)

    return models, optims


def get_model_real(opt, device):
    import models.conv_lstms as lstm_models
    import models.mask_predictor as mask_predictor_model

    if opt.dataset == 'kitti':
        import models.kitti_models as model
    else:
        import models.cityscapes_models as model
    # lstms
    models = {}
    im_size = (4, 4) if opt.dataset == 'kitti' else (4, 8)
    static_frame_predictor = lstm_models.conv_lstm(opt.g_dim_app + opt.z_dim_app, opt.g_dim_app, opt.rnn_size,
                                                   opt.predictor_rnn_layers, opt.batch_size, im_size, device)
    dynamic_frame_predictor = lstm_models.conv_lstm(opt.g_dim_motion + opt.z_dim_motion, opt.g_dim_motion,
                                                    opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size, im_size,
                                                    device)

    posterior_app = lstm_models.gaussian_conv_lstm(opt.g_dim_app, opt.z_dim_app, opt.rnn_size,
                                                   opt.posterior_rnn_layers, opt.batch_size, im_size, device)
    prior_app = lstm_models.gaussian_conv_lstm(opt.g_dim_app, opt.z_dim_app, opt.rnn_size,
                                               opt.prior_rnn_layers, opt.batch_size, im_size, device)

    posterior_motion = lstm_models.gaussian_conv_lstm(opt.g_dim_motion, opt.z_dim_motion, opt.rnn_size,
                                                      opt.posterior_rnn_layers, opt.batch_size, im_size, device)
    prior_motion = lstm_models.gaussian_conv_lstm(opt.g_dim_motion, opt.z_dim_motion, opt.rnn_size,
                                                  opt.prior_rnn_layers, opt.batch_size, im_size, device)
    # convs
    assert opt.g_dim_app == opt.g_dim_motion, "Motion and Appearance feature sizes should be the same."
    img_encoder = model.encoder(opt.g_dim_app * 2, opt.channels)
    pixel_encoder = model.spatial_encoder(opt.g_dim_app, opt.g_dim_app * 2)
    motion_encoder = model.spatial_encoder(opt.g_dim_motion, opt.g_dim_motion * 4)

    pixel_decoder = model.decoder(opt.g_dim_app, opt.channels)  # 3 channels for RGB, 1 channel for mnist
    flow_decoder = model.decoder(opt.g_dim_motion, 2, act=None)  # 2 channel flow, x and y
    mask_decoder = mask_predictor_model.mask_predictor(opt.channels * 2)

    img_encoder.apply(utils.init_weights)
    pixel_encoder.apply(utils.init_weights)
    motion_encoder.apply(utils.init_weights)

    pixel_decoder.apply(utils.init_weights)
    flow_decoder.apply(utils.init_weights)
    mask_decoder.apply(utils.init_weights)

    models['static_frame_predictor'] = static_frame_predictor.to(device)
    models['dynamic_frame_predictor'] = dynamic_frame_predictor.to(device)
    models['posterior_app'] = posterior_app.to(device)
    models['prior_app'] = prior_app.to(device)
    models['posterior_motion'] = posterior_motion.to(device)
    models['prior_motion'] = prior_motion.to(device)

    models['img_encoder'] = img_encoder.to(device)
    models['pixel_encoder'] = pixel_encoder.to(device)
    models['motion_encoder'] = motion_encoder.to(device)

    models['pixel_decoder'] = pixel_decoder.to(device)
    models['flow_decoder'] = flow_decoder.to(device)
    models['mask_decoder'] = mask_decoder.to(device)

    optims = get_optimizers(models, opt)

    return models, optims


def get_optimizers(models, opt):
    return {key: opt.optimizer(val.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) for key, val in models.items()}


def load_model_from_checkpoint(opt, device):
    saved_model = torch.load(opt.model_path)
    model = {}
    for key in saved_model.keys():
        if any(model_type in key for model_type in ['encoder', 'decoder', 'prior', 'posterior', 'predictor']):
            model[key] = saved_model[key].to(device)
            model[key].train()
            if any(model_type in key for model_type in ['prior', 'posterior', 'predictor']):
                model[key].batch_size = opt.batch_size
                model[key].eval()
    try:
        opt.is_real_dataset = False
        tmp_opt = vars(saved_model['opt'])  # if it is a generic dataset, it will be a namespace object
    except:
        opt.is_real_dataset = True
        tmp_opt = saved_model['opt']
    opt.last_frame_skip = tmp_opt['last_frame_skip']
    opt.channels = tmp_opt['channels']
    opt.running_avg = tmp_opt['running_avg']
    opt.motion_skip = tmp_opt['motion_skip']
    opt.dataset = tmp_opt['dataset']
    opt.num_digits = tmp_opt['num_digits']
    if opt.dataset in ['kitti', 'cityscapes']:
        opt.is_real_dataset = True
    else:
        opt.is_real_dataset = False
    return model


def kl_criterion(mu1, logvar1, mu2, logvar2, bs):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) = 
    #   log( sqrt(
    # 
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
    return kld.sum() / bs


def train_step(x, models, optims, opt, device):
    for key, val in models.items():
        val.zero_grad()
    for key, val in models.items():
        if any(model_type in key for model_type in ['prior', 'posterior', 'predictor']):
            # if an lstm-variant
            val.hidden = val.init_hidden(device)

    pixel_mse = flow_mse = mask_mse = kld = 0

    if opt.is_real_dataset:
        img_encoder = models['img_encoder']
    pixel_encoder = models['pixel_encoder']
    motion_encoder = models['motion_encoder']
    static_frame_predictor = models['static_frame_predictor']
    dynamic_frame_predictor = models['dynamic_frame_predictor']
    posterior_app = models['posterior_app']
    prior_app = models['prior_app']
    posterior_motion = models['posterior_motion']
    prior_motion = models['prior_motion']
    pixel_decoder = models['pixel_decoder']
    flow_decoder = models['flow_decoder']
    mask_decoder = models['mask_decoder']
    x_in = x[0]
    N = (opt.n_past + opt.n_future)
    for i in range(1, N):
        if opt.is_real_dataset:
            encoded_current_img, skip = img_encoder(x_in)
            encoded_target_img, _ = img_encoder(x[i])
            h = pixel_encoder(encoded_current_img)  # features of current frame
            h_target = pixel_encoder(encoded_target_img)  # features of next frame
            h_motion = motion_encoder(
                torch.cat([encoded_current_img, encoded_target_img], dim=1))  # motion freature from t-1 -> t
            if i == 1:
                # for t == 0, there is no motion. we want model to learn this no motion.
                last_motion = motion_encoder(torch.cat([encoded_current_img, encoded_current_img], dim=1))
            skip_motion = None
        else:
            h, skip = pixel_encoder(x_in)  # features of current frame
            h_target = pixel_encoder(x[i])[0]  # features of next frame
            h_motion, skip_motion = motion_encoder(torch.cat([x_in, x[i]], dim=1))  # motion freature from t-1 -> t
            if i == 1:
                # for t == 0, there is no motion. we want model to learn this no motion.
                last_motion, last_motion_skip = motion_encoder(torch.cat([x[0], x[0]], dim=1))

        if opt.last_frame_skip or i < opt.n_past:
            # h, skip = h
            if opt.running_avg:
                if i == 1:
                    skips = skip
                else:
                    for idx in range(len(skips)):
                        skips[idx] = (skips[idx] * (i - 1) + skip[idx]) / i
            else:
                skips = skip
        else:
            h = h[0]

        z_t, mu, logvar = posterior_app(h_target)  # posterior from pixel
        _, mu_p, logvar_p = prior_app(h)  # prior from pixel

        z_t_motion, mu_posterior_motion, logvar_posterior_motion = posterior_motion(h_motion)  # posterior from motion
        _, mu_prior_motion, logvar_prior_motion = prior_motion(last_motion)  # prior from motion

        h_pred_static = static_frame_predictor(torch.cat([h, z_t], dim=1))  # prediction from pixel branch
        h_pred_dynamic = dynamic_frame_predictor(
            torch.cat([last_motion, z_t_motion], dim=1))  # prediction from motion branch

        x_pred = pixel_decoder([h_pred_static, skips])  # get pixel image from the h_pred

        if opt.motion_skip:
            # flow prediction with motion encoder's skip connection
            flow_pred = flow_decoder([h_pred_dynamic, last_motion_skip])
        else:
            flow_pred = flow_decoder([h_pred_dynamic, skips])  # flow prediction with pixel encoder's skip connection

        warped_x_pred = torch.clamp(warp(x_in, flow_pred), min=0, max=1)  # warp current to next image

        mask_pred = mask_decoder(torch.cat([x_pred, warped_x_pred], dim=1))

        pred_x = mask_pred * x_pred + (1 - mask_pred) * warped_x_pred  # get the final output by weighting

        last_motion = h_motion
        if not opt.is_real_dataset:
            if not skip_motion:
                raise Exception("skip_motion is not assigned.")
            last_motion_skip = skip_motion
        # losses
        pixel_mse += F.mse_loss(x_pred, x[i])
        flow_mse += F.mse_loss(warped_x_pred, x[i])
        mask_mse += F.mse_loss(pred_x, x[i])
        kld += kl_criterion(mu, logvar, mu_p, logvar_p, opt.batch_size)
        kld += kl_criterion(mu_posterior_motion, logvar_posterior_motion,
                            mu_prior_motion, logvar_prior_motion, opt.batch_size)
        if opt.sch_sampling != 0:
            # can do scheduled sampling
            if i < opt.n_past:
                x_in = x[i]
                continue
            val = random.random()
            if val < opt.sc_prob:
                # use gt
                x_in = x[i]
            else:
                # use prediction
                x_in = pred_x.clone()
                x_in = x_in.detach()
        else:
            x_in = x[i]

    loss = pixel_mse + flow_mse + mask_mse + kld * opt.beta
    loss.backward()

    for optimizer in optims.values():
        optimizer.step()

    pixel_mse = pixel_mse.data.cpu().numpy() / N
    flow_mse = flow_mse.data.cpu().numpy() / N
    mask_mse = mask_mse.data.cpu().numpy() / N
    kld = kld.data.cpu().numpy() / N

    return pixel_mse, flow_mse, mask_mse, kld


def eval_step(x, models, opt, device, lpips_model):
    # get approx posterior sample
    nsample = opt.nsample
    metric_best = {}
    sample_best = {}

    if opt.is_real_dataset:
        img_encoder = models['img_encoder']
    pixel_encoder = models['pixel_encoder']
    motion_encoder = models['motion_encoder']
    static_frame_predictor = models['static_frame_predictor']
    dynamic_frame_predictor = models['dynamic_frame_predictor']
    posterior_app = models['posterior_app']
    prior_app = models['prior_app']
    posterior_motion = models['posterior_motion']
    prior_motion = models['prior_motion']
    pixel_decoder = models['pixel_decoder']
    flow_decoder = models['flow_decoder']
    mask_decoder = models['mask_decoder']

    with torch.no_grad():
        for s in range(nsample):
            gen_seq = []
            gt_seq = []

            static_frame_predictor.hidden = static_frame_predictor.init_hidden(device)
            dynamic_frame_predictor.hidden = dynamic_frame_predictor.init_hidden(device)
            posterior_app.hidden = posterior_app.init_hidden(device)
            prior_app.hidden = prior_app.init_hidden(device)
            posterior_motion.hidden = posterior_motion.init_hidden(device)
            prior_motion.hidden = prior_motion.init_hidden(device)

            x_in = x[0]
            for i in range(1, opt.n_eval):
                if opt.is_real_dataset:
                    encoded_current_img, skip = img_encoder(x_in)
                    h = pixel_encoder(encoded_current_img)
                else:
                    h, skip = pixel_encoder(x_in)
                if opt.last_frame_skip or i < opt.n_past:
                    # h, skip = h
                    if opt.running_avg:
                        if i == 1:
                            skips = skip
                        else:
                            skips = [(skips[idx] * (i - 1) + skip[idx]) / i for idx in range(len(skips))]
                    else:
                        skips = skip
                else:
                    h, _ = h
                h = h.detach()
                if i < opt.n_past:
                    # we have access to GT
                    if opt.is_real_dataset:
                        encoded_target_img, _ = img_encoder(x[i])
                        h_target = pixel_encoder(encoded_target_img).detach()
                        h_motion = motion_encoder(torch.cat([encoded_current_img, encoded_target_img], dim=1))
                        if i == 1:
                            # for t == 0, there is no motion. we want model to learn this no motion.
                            last_motion = motion_encoder(torch.cat([encoded_current_img, encoded_current_img], dim=1))
                    else:
                        h_target = pixel_encoder(x[i])[0].detach()
                        h_motion, motion_skip = motion_encoder(torch.cat([x[i - 1], x[i]], dim=1))
                        if i == 1:
                            # for t == 0, there is no motion. we want model to learn this no motion.
                            last_motion, last_motion_skip = motion_encoder(torch.cat([x[0], x[0]], dim=1))

                    z_t, _, _ = posterior_app(h_target)
                    prior_app(h)
                    z_t_motion, _, _ = posterior_motion(h_motion)
                    prior_motion(last_motion)

                    static_frame_predictor(torch.cat([h, z_t], dim=1))
                    dynamic_frame_predictor(torch.cat([last_motion, z_t_motion], dim=1))
                    x_in = x[i]
                    last_motion = h_motion
                    if not opt.is_real_dataset:
                        last_motion_skip = motion_skip
                else:
                    # we dont have access to GT
                    z_t, _, _ = prior_app(h)
                    z_t_motion, _, _ = prior_motion(last_motion)

                    h_pred_static = static_frame_predictor(torch.cat([h, z_t], dim=1))
                    h_pred_dynamic = dynamic_frame_predictor(torch.cat([last_motion, z_t_motion], dim=1))

                    x_pred = pixel_decoder([h_pred_static, skips])  # get pixel image from the h_pred

                    if opt.motion_skip:
                        # flow prediction with motion encoder's skip connection
                        flow_pred = flow_decoder([h_pred_dynamic, last_motion_skip])
                    else:
                        # flow prediction with pixel encoder's skip connection
                        flow_pred = flow_decoder([h_pred_dynamic, skips])

                    warped_x_pred = torch.clamp(warp(x_in, flow_pred), min=0, max=1)  # warp current to next image

                    mask_pred = mask_decoder(torch.cat([x_pred, warped_x_pred], dim=1))

                    last_frame = x_in
                    x_in = mask_pred * x_pred + (1 - mask_pred) * warped_x_pred  # get the final output by weighting
                    if opt.is_real_dataset:
                        encoded_target_img, _ = img_encoder(x_in)
                        last_motion = motion_encoder(torch.cat([encoded_current_img, encoded_target_img], dim=1))
                    else:
                        last_motion, last_motion_skip = motion_encoder(torch.cat([last_frame, x_in], dim=1))

                    gen_seq.append(x_in.data.cpu().numpy())
                    gt_seq.append(x[i].data.cpu().numpy())

            gen_seq = torch.from_numpy(np.stack(gen_seq))
            gt_seq = torch.from_numpy(np.stack(gt_seq))
            ssim_score = metric_helpers._ssim_wrapper(gen_seq, gt_seq).mean(2).mean(0)
            mse = torch.mean(F.mse_loss(gen_seq, gt_seq, reduction='none'), dim=[3, 4])
            pnsr_score = 10 * torch.log10(1 / mse).mean(2).mean(0).cpu()
            lpips_score = metric_helpers._lpips_wrapper(gen_seq, gt_seq, lpips_model).mean(0).cpu()
            results = {
                'psnr': pnsr_score,
                'ssim': ssim_score,
                'lpips': lpips_score
            }
            pred = gen_seq.cpu().permute(1, 0, 3, 4, 2)
            for name, val in results.items():
                if s == 0:
                    metric_best[name] = val.clone()  # Metric value for the current best prediction
                    sample_best[name] = pred.clone()
                else:
                    idx_better = metric_helpers._get_idx_better(name, metric_best[name], val)
                    metric_best[name][idx_better] = val[idx_better]
                    sample_best[name][idx_better] = pred[idx_better]

    return sample_best, metric_best


def eval_model(val_loader, models, opt, device):
    all_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    all_samples = {'psnr': [], 'ssim': [], 'lpips': []}
    for val_batch in val_loader:
        val_batch = val_batch.to(device)
        samples, metrics = eval_step(val_batch, models, opt, device)
        for name in samples.keys():
            all_metrics[name] += metrics[name].cpu().detach().numpy().tolist()
            all_samples[name] += [
                (samples[name].cpu().detach().numpy() * 255).astype('uint8')]  # shape is different now, check this
    metrics_return = {}
    samples_return = {}
    for name, values in all_metrics.items():
        metrics_return[name] = np.mean(values)
        samples_return[name] = np.transpose(np.concatenate(all_samples[name], axis=0), (1, 0, 4, 2, 3))
    return metrics_return, samples_return
