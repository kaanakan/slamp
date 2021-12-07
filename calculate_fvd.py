import os
import sys

import numpy as np
import torch

from metrics.fvd.score import fvd as fvd_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = '3'

path = sys.argv[1]
name = sys.argv[2]

gts = np.load(os.path.join(path, 'gts.npz'))['samples']
preds = np.load(os.path.join(path, name))['samples'] / 255.
# preds = np.transpose(preds, (1, 0, 4, 2, 3))
# shape preds = batch, length, w, h, c

print(gts.shape, gts.min(), gts.max())
print(preds.shape, preds.min(), preds.max())

n_past = gts.shape[0] - preds.shape[0]

preds_with_gt = torch.from_numpy(np.concatenate([gts[:n_past], preds], 0)).float()
gts = torch.from_numpy(gts)
print(preds_with_gt.shape, preds_with_gt.min(), preds_with_gt.max())
print(gts.shape, gts.min(), gts.max())

fvd_value = fvd_score(gts, preds_with_gt)
print("FVD SCORE: ", fvd_value)
