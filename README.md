# Artist Style Transfer Via Quadratic Potential

[**Rahul Bhalley**](https://github.com/rahulbhalley) and [Jianlin Su](https://github.com/bojone)

[arXiv paper](https://arxiv.org/abs/1902.11108)

### Abstract

In this paper we address the problem of artist style transfer where the painting style of a given artist is applied on a real world photograph. We train our neural networks in adversarial setting via recently introduced quadratic potential divergence for stable learning process. To further improve the quality of generated artist stylized images we also integrate some of the recently introduced deep learning techniques in our method. To our best knowledge this is the first attempt towards artist style transfer via quadratic potential divergence. We provide some stylized image samples in the supplementary material. The source code for experimentation was written in [PyTorch](https://pytorch.org) and is available online in my [GitHub repository](https://github.com/rahulbhalley/cyclegan-qp).

If you find our work, or this repository helpful, please consider citing our work with the following BibTex:
```
@article{bhalley2019artist,
  title={Artist Style Transfer Via Quadratic Potential},
  author={Bhalley, Rahul and Su, Jianlin},
  journal={arXiv preprint arXiv:1902.11108},
  year={2019}
}
```

**NOTE: Pre-trained models are available in [`Google Drive`](https://drive.google.com/drive/folders/1IJ0OswLFD_-P2wgDg6RJhf29AQxYUc0_?usp=sharing). Please download it in the root directory of this repository.**

### Prerequisites

This code was tested in following environment setting:

- Python (version >= 3.6.0)
- [PyTorch](https://github.com/pytorch/pytorch) (version >= 1.0.0)
- [Torchvision](https://github.com/pytorch/vision) (version = 0.2.1)

### Usage

First clone this repository:
```
git clone https://github.com/rahulbhalley/cyclegan-qp.git
```

#### Getting Datasets

Enter into the `cyclegan-qp` directory via terminal.
```
cd cyclegan-qp
```

To download the datasets (for instance, `ukiyoe2photo`) run:
```
bash download_dataset.sh ukiyoe2photo
```

Now `ukiyoe2photo` dataset will be downloaded and unzipped in `cyclegan-qp/datasets/ukiyoe2photo` directory.

#### Training & Inference

To train the network set `TRAIN = True` in [config.py](https://github.com/rahulbhalley/cyclegan-qp/blob/main/config.py) and for inference set it to `False`. Then one may only need to execute the following command in terminal.
```
python main.py
```

#### Configurations

Following is a list of configurable variables (in [config.py](https://github.com/rahulbhalley/cyclegan-qp/blob/main/config.py)) to perform experiments with different settings.

##### Data

- `DATASET_DIR` - name of directory containing dataset. Default: `"datasets"`.
- `DATASET_NAME` - name of dataset to use. Default: `"vangogh2photo"`.
- `LOAD_DIM` - sets the size of images to load. Default: `286`.
- `CROP_DIM` - square crops the images from center. Default: `256`.
- `CKPT_DIR` - name of directory to save checkpoints in. Default: `"checkpoints"`.
- `SAMPLE_DIR` - directory name where inferred samples will be saved. Default: `"samples"`.

##### Quadratic Potential

- `LAMBDA` - see equation (1) in [paper](https://arxiv.org/abs/1902.11108). Default: `10.0`.
- `NORM` - see equation (2) in [paper](https://arxiv.org/abs/1902.11108). Possible values: `"l1"`, `"l2"`. Default: `"l1"`.

##### CycleGAN-QP

- `CYC_WEIGHT` - cycle consistency weight. Default: `10.0`.
- `ID_WEIGHT` - identity weight. Default: `0.5`.

##### Network

- `N_CHANNELS` - number of channels of images in dataset. Set to `3` for RGB and `1` for grayscale. Default: `3`.
- `UPSAMPLE` - set `True` to use ([Odena et al., 2016](https://distill.pub/2016/deconv-checkerboard/)) technique but `False` to use vanilla transpose convolution layers in generator networks. Default: `True`.

##### Training

- `RANDOM_SEED` - random seed to reproduce the experiments. Default: `12345`.
- `BATCH_SIZE` - batch size for training. Default: `4`.
- `LR` - learning rate. Default: `2e-4`.
- `BETA1` - hyper-parameter of Adam optimizer. Default: `0.5`.
- `BETA2` - hyper-parameter of Adam optimizer. Default: `0.999`.
- `BEGIN_ITER` - if `0` the train begins from start but when set to `> 0` then training continues from `BEGIN_ITER`th checkpoint. Default: `0`.
- `END_ITER` - number of iteration for training. Default: `15000`.
- `TRAIN` - set `True` for training CycleGAN-QP but `False` to perform inference (for more inference configurations see next subsection). Default: `True`.

##### Inference

- `INFER_ITER` - performs inference by loading parameters from this checkpoint. Default: `15000`.
- `INFER_STYLE` - style to be transferred on images. Possible values: `"ce"`, `"mo"`, `"uk"`, `"vg"`. Default: `"vg"`.
- `IMG_NAME` - name of image to be performed inference on. Default: `"image.jpg"`.
- `IN_IMG_DIR` - name of directory containing `IMG_NAME`. Default: `"images"`.
- `OUT_STY_DIR` - name of directory to save inferred `IMG_NAME`. Default: `"sty"`.
- `OUT_REC_DIR` - name of directory to save recovered (original) `IMG_NAME`. Default: `"rec"`.
- `IMG_SIZE` - set `None` to infer with the original sized `IMG_NAME` or set some integral value to infer with `IMG_SIZE`. Default: `None`.

##### Logs

- `ITERS_PER_LOG` - iterations duration at which screen logs should be made. Default: `100`
- `ITERS_PER_CKPT` - iterations duration at which checkpoints should be saved. Default: `1000`

### Results

The images in each column (from left to right) corresponds to:
- Original image
- Paul Cézanne
- Claude Monet
- Ukiyo-e
- Vincent Van Gogh. 

And each row contains a different image.

#### Real Image to Stylized Image
![](https://github.com/rahulbhalley/cyclegan-qp/raw/main/assets/grid_sty.jpg)

#### Stylized Image to Real Image
![](https://github.com/rahulbhalley/cyclegan-qp/raw/main/assets/grid_rec.jpg)
