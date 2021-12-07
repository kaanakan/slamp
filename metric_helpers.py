import torch

from metrics.ssim import ssim_loss


def _get_idx_better(name, ref, hyp):
    """
    Computes the batch indices for which the input metric value is better than the current metric value.
    Parameters
    ----------
    name : str
        'mse', 'psnr', 'ssim', 'lpips', or 'fvd'. Metric to consider. For 'mse', 'fvd' and 'lpips', lower is better,
        while for 'psnr' and 'ssim', higher is better.
    ref : torch.*.Tensor
        One-dimensional tensor containing a list of current metric values.
    hyp : torch.*.Tensor
        One-dimensional tensor containing a list of new metric values to be compared agains ref. Must be of the same
        length as ref.
    Returns
    -------
    torch.*.LongTensor
        List of indices i for which the value hyp[i] is better than the value ref[i].
    """
    if name in ('mse', 'fvd', 'lpips'):
        return torch.nonzero(hyp < ref, as_tuple=False).flatten()
    if name in ('psnr', 'ssim'):
        return torch.nonzero(hyp > ref, as_tuple=False).flatten()
    raise ValueError(f'Metric \'{name}\' not yet implemented')


def _ssim_wrapper(sample, gt):
    """
    Computes the pixel-averaged SSIM between two videos.
    Parameters
    ----------
    sample : torch.*.Tensor
        Tensor representing a video, of shape (length, batch, channels, width, height) and with float values lying in
        [0, 1].
    gt : torch.*.Tensor
        Tensor representing a video, of shape (length, batch, channels, width, height) and with float values lying in
        [0, 1]. Its shape should be the same as sample.
    Returns
    -------
    torch.*.Tensor
        Tensor of pixel-averaged SSIM between the input videos, of shape (length, batch, channels).
    """
    nt, bsz = sample.shape[0], sample.shape[1]
    img_shape = sample.shape[2:]
    ssim = ssim_loss(sample.view(nt * bsz, *img_shape), gt.view(nt * bsz, *img_shape), max_val=1., reduction='none')
    return ssim.mean(dim=[2, 3]).view(nt, bsz, img_shape[0])


def _lpips_wrapper(sample, gt, lpips_model):
    """
    Computes the frame-wise LPIPS between two videos.
    Parameters
    ----------
    sample : torch.*.Tensor
        Tensor representing a video, of shape (length, batch, channels, width, height) and with float values lying in
        [0, 1].
    gt : torch.*.Tensor
        Tensor representing a video, of shape (length, batch, channels, width, height) and with float values lying in
        [0, 1]. Its shape should be the same as sample.
    Returns
    -------
    torch.*.Tensor
        Tensor of frame-wise LPIPS between the input videos, of shape (length, batch).
    """
    nt, bsz = sample.shape[0], sample.shape[1]
    img_shape = sample.shape[2:]
    # Switch to three color channels for grayscale videos
    if img_shape[0] == 1:
        sample_ = sample.repeat(1, 1, 3, 1, 1)
        gt_ = gt.repeat(1, 1, 3, 1, 1)
    else:
        sample_ = sample
        gt_ = gt
    lpips = lpips_model(sample_.view(nt * bsz, 3, *img_shape[1:]), gt_.view(nt * bsz, 3, *img_shape[1:]))
    return lpips.view(nt, bsz)
