# SLAMP: Stochastic Latent Appearance and Motion Prediction

Official implementation of the paper *SLAMP: Stochastic Latent Appearance and Motion Prediction* (Adil Kaan Akan, Erkut Erdem, Aykut Erdem, Fatma Guney), accepted and presented at ICCV 2021.


## [Article](https://openaccess.thecvf.com/content/ICCV2021/html/Akan_SLAMP_Stochastic_Latent_Appearance_and_Motion_Prediction_ICCV_2021_paper.html)


## [Preprint](http://arxiv.org/abs/2108.02760)


## [Project Website](https://kuis-ai.github.io/slamp/)


## [Pretrained Models]()


## Requirements

All models were trained with Python 3.7.6 and PyTorch 1.4.0 using CUDA 10.1.

A list of required Python packages is available in the `requirements.txt` file.



## Datasets

For preparations of datasets, we followed [SRVP's code](https://github.com/edouardelasalles/srvp). Please follow the links below if you want to construct the datasets.

[Stochastic Moving MNIST](https://github.com/edouardelasalles/srvp#stochastic-moving-mnist)

[KTH](https://github.com/edouardelasalles/srvp#kth)

[BAIR](https://github.com/edouardelasalles/srvp#bair)

### KITTI

For KITTI, you need to download the Raw KITTI dataset and extract the zip files. You can follow the official [KITTI page](http://www.cvlibs.net/datasets/kitti/raw_data.php). 

A good idea might be preprocessing every image in the dataset so that all of them have a size of (w=310, h=92). Then, you can disable the resizing operation in the data loaders, which will speed up the training.

### Cityscapes

For Cityscapes, you need to download `leftImg8bit_sequence` from the official [Cityscapes page](https://www.cityscapes-dataset.com/).

`leftImg8bit_sequence` contains 30-frame snippets (17Hz) surrounding each left 8-bit image (-19 | +10) from the train, val, and test sets (150000 images).

A good idea might be preprocessing every image in the dataset so that all of them have a size of (w=256, h=128). Then, you can disable the resizing operation in the data loaders, which will speed up the training.

## Training

To train a new model, the script `train.py` should be used as follows: 

Data directory (`$DATA_DIR`) and `$SAVE_DIR` must be given using options `--data_root $DATA_DIR --log_dir $SAVE_DIR`. To use GPU, you need to use `--device` flag.

- for Stochastic Moving MNIST:
```
--n_past 5 --n_future 10 --n_eval 25 --z_dim_app 20 --g_dim_app 128 --z_dim_motion 20
--g_dim_motion 128 --last_frame_skip --running_avg --batch_size 32
```

- for KTH:
```
--dataset kth --n_past 10 --n_future 10 --n_eval 40 --z_dim_app 50 --g_dim_app 128 --z_dim_motion 50 --model vgg
--g_dim_motion 128 --last_frame_skip --running_avg --sch_sampling 25 --batch_size 20
```

- for BAIR:
```
--dataset bair --n_past 2 --n_future 10 --n_eval 30 --z_dim_app 64 --g_dim_app 128 --z_dim_motion 64 --model vgg
--g_dim_motion 128 --last_frame_skip --running_avg --sch_sampling 25 --batch_size 20 --channels 3
```

- for KITTI:
```
--dataset bair --n_past 10 --n_future 10 --n_eval 30 --z_dim_app 32 --g_dim_app 64 --z_dim_motion 32 --batch_size 8
--g_dim_motion 64 --last_frame_skip --running_avg --model vgg --niter 151 --channels 3
```

- for Cityscapes:
```
--dataset bair --n_past 10 --n_future 10 --n_eval 30 --z_dim_app 32 --g_dim_app 64 --z_dim_motion 32 --batch_size 7
--g_dim_motion 64 --last_frame_skip --running_avg --model vgg --niter 151 --channels 3 --epoch_size 1300
```



## Testing

To evaluate a trained model, the script `evaluate.py` should be used as follows:
```bash
python evaluate.py --data_root $DATADIR --log_dir $LOG_DIR --model_path $MODEL_PATH
```
where `$LOG_DIR` is a directory where the results will be saved, `$DATADIR` is the directory containing the test set.

**Important note:** The directory containing the script should include a directory called `lpips_weights` which contains [v0.1 LPIPS weights](https://github.com/richzhang/PerceptualSimilarity/tree/master/lpips/weights) (from the official repository of [*The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*](https://github.com/richzhang/PerceptualSimilarity)).

To run the evaluation on GPU, use the option `--device`.


<details>
<summary>Pretrained weight links with Dropbox</summary>
- For MNIST:

```bash
wget https://www.dropbox.com/s/eseisehe2u0epiy/slamp_mnist.pth
```

- For KTH:

```bash
wget https://www.dropbox.com/s/7m0806nt7xt9bz8/slamp_kth.pth
```

- For BAIR:

```bash
wget https://www.dropbox.com/s/cl1pzs5trw3ltr0/slamp_bair.pth
```

- For KITTI:

```bash
wget https://www.dropbox.com/s/p7wdboswakyj7yi/slamp_kitti.pth
```

- For Cityscapes:

```bash
wget https://www.dropbox.com/s/lzwiivr1irffhsj/slamp_cityscapes.pth
```
</details>


PSNR, SSIM, and LPIPS results reported in the paper were obtained with the following options:
- for stochastic Moving MNIST:

```bash
python evaluate.py --data_root $DATADIR --log_dir $LOG_DIR --model_path $MODEL_PATH --n_past 5 --n_future 20
```

- for KTH:
```bash
python evaluate.py --data_root $DATADIR --log_dir $LOG_DIR --model_path $MODEL_PATH --n_past 10 --n_future 30
```

- for BAIR:
```bash
python evaluate.py --data_root $DATADIR --log_dir $LOG_DIR --model_path $MODEL_PATH --n_past 2 --n_future 28
```

- for KITTI:
```bash
python evaluate.py --data_root $DATADIR --log_dir $LOG_DIR --model_path $MODEL_PATH --n_past 10 --n_future 20
```

- for Cityscapes:
```bash
python evaluate.py --data_root $DATADIR --log_dir $LOG_DIR --model_path $MODEL_PATH --n_past 10 --n_future 20
```

To calculate FVD results, you can use `calculate_fvd.py` script as follows:
```
python calculate_fvd.py $LOG_DIR $SAMPLE_NAME
```
where `$LOG_DIR` is the directory containg the results generated by the evaluate script and `$SAMPLE_NAME` is the file which contains the samples such as `psnr.npz`, `ssim.npz` or `lpips.npz`. The script will print the FVD value at the end.

## How to Cite

Please cite the paper if you benefit from our paper or the repository:

```
@InProceedings{Akan2021ICCV,
    author    = {Akan, Adil Kaan and Erdem, Erkut and Erdem, Aykut and Guney, Fatma},
    title     = {SLAMP: Stochastic Latent Appearance and Motion Prediction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14728-14737}
}
```

## Acknowledgments

We want to thank SRVP and SVG authors for making their repositories public. This repository contains several code segments from [SRVP's repository](https://github.com/edouardelasalles/srvp) and [SVG's repository](https://github.com/edenton/svg). We appreciate the efforts by [Berkay Ugur Senocak](https://4turkuaz.github.io/) for cleaning the code before release.
