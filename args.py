import argparse

import torch.optim as optim


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--log_dir', default='logs/lp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--name', default='', help='identifier for directory')
    parser.add_argument('--data_root', default='data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--epoch_size', type=int, default=1000, help='epoch size')
    parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--channels', default=1, type=int)
    parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
    parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
    parser.add_argument('--n_eval', type=int, default=25, help='number of frames to predict during eval')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim_app', type=int, default=20, help='dimensionality of z_t')
    parser.add_argument('--g_dim_app', type=int, default=128,
                        help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--z_dim_motion', type=int, default=20, help='dimensionality of z_t')
    parser.add_argument('--g_dim_motion', type=int, default=128,
                        help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
    parser.add_argument('--data_threads', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
    parser.add_argument('--last_frame_skip', default=True, action='store_true',
                        help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--device', action='store_true', help='if true, gpu will be used.')
    parser.add_argument('--running_avg', default=True, action='store_true',
                        help='if true, running average of skip connections will be used')
    parser.add_argument('--sch_sampling', type=int, default=0,
                        help='if given an integer, scheduled sampling will be used. inverse sigmoid with k.')
    parser.add_argument('--two_lstm', action='store_true', help='if used, static and dynamic lstms will be used.')
    parser.add_argument('--motion_skip', action='store_true',
                        help='if used, motion encoder\'s skip connections will be used in flow decoder.')
    opt = parser.parse_args()
    if opt.optimizer == 'adam':
        opt.optimizer = optim.Adam
    elif opt.optimizer == 'rmsprop':
        opt.optimizer = optim.RMSprop
    elif opt.optimizer == 'sgd':
        opt.optimizer = optim.SGD
    opt.sc_prob = 1
    name = 'model=%s-rnn_size=%d-rnn_layers=%d-%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim_app=%d-z_dim_app=%d-g' \
           '_dim_motion=%d-z_dim_motion=%d-last_frame_skip=%s_running_avg=%s_sch_sampling=%d_two_lstm=%s_motion_skip' \
           '=%s-beta=%.7f%s' % (opt.model, opt.rnn_size, opt.predictor_rnn_layers, opt.posterior_rnn_layers,
                                opt.prior_rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim_app, opt.z_dim_app,
                                opt.g_dim_motion, opt.z_dim_motion, opt.last_frame_skip, opt.running_avg,
                                opt.sch_sampling, opt.two_lstm, opt.motion_skip, opt.beta, opt.name)

    opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name) \
        if opt.dataset == 'smmnist' else '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

    opt.is_real_dataset = True if opt.dataset in ['kitti', 'cityscapes'] else False
    return opt
