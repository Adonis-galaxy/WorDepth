# WorDepth: Variational Language Prior for Monocular Depth Estimation #

Official implementation of the paper "WorDepth: Variational Language Prior for Monocular Depth Estimation"

Accepted by CVPR 2024

Paper Link: https://arxiv.org/abs/2404.03635

Presentation Video (5min): https://www.youtube.com/watch?v=QNwOFZZc8XI

Authors: Ziyao Zeng, Daniel Wang, Fengyu Yang, Hyoungseob Park, Yangchao Wu, Stefano Soatto, Byung-Woo Hong, Dong Lao, Alex Wong

## Overview ##
3D reconstruction from a single image is an ill-posed problem, since there exists infinitely many 3D scenes, with different scales, that can generate an image. For example,  one bed can be smaller and put closer, or can be bigger and put further. They look the same in the same image by projection.

Also, 3D reconstruction from a text caption is also an ill-posed problem, since there exists infinitely many 3D scenes that fits a description. For example, for “a bedroom with a stand and a bed”, the stand and the bed can be anywhere in the bedroom.

So here comes the question: Can two modalities that are inherently ambiguous, single image and text caption, resolve one another’s ambiguity in 3D reconstruction?

And our Key idea is Using language to ground depth estimates to metric scale!
We do so simply by letting the model know what objects are around and it can better estimate scale.

<img src="figures/teaser.png" alt="teaser" width="500"/>

### Pipeline ###
We train a text-VAE to encode text into the mean and standard deviation parameterizing the distribution of 3D scenes for a description.
For one text caption, we encode the mean and standard deviation, then sample a feature from the Gaussian with such mean and standard deviation to generate a depth map that aligned with such a description.

Then, in inference, we choose one of the infinitely many scenes matching the description that is compatible with the observed image using a Conditional Sampler. We do so by predict epsilon tilt in the reparameterizaion step instead of sampling it from a standard Gaussian. And we alternatively optimize the text-VAE and the Conditional Sampler with a ratio p.
![pipeline](figures/pipeline.png)

### Visualization on NYU-Depth-v2 ###
Knowing that certain objects and that they are typically of certain sizes exist in the scene, we can better estimate the scale as evident by the uniform improvement over the error maps.

![vis_nyu](figures/vis_nyu.png)

### Poster ###
![poster](figures/poster.png)
## Setup Environment ##
Create Virtual Environment:
```
virtualenv -p /usr/bin/python3.8 ~/venvs/wordepth

vim  ~/.bash_profile
```
Insert the following line to vim:
```
alias wordepth="export CUDA_HOME=/usr/local/cuda-11.1 && source ~/venvs/wordepth/bin/activate"
```
Then activate it, install all packages:
```
source ~/.bash_profile

wordepth

pip install -r requirements.txt
```

Download Swin-L checkpoint
```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
```

For simplicity, we have extracted image caption using [ExpansionNet v2](https://github.com/jchenghu/ExpansionNet_v2) in "./text" and extracted their CLIP text features in "./text_feat". While running, the model will automatically load text features.

## Setup Datasets ##

### Run NYU-Depth-v2 ###
Specify GPU Number train_nyu.sh, then run by:
```
sh train_nyu.sh
```
Before running new experiments, remember to change the model_name in train_nyu.sh and config/arguments_train_nyu.txt to be the same.

### Run KITTI ###
Specify GPU Number train_kitti.sh, then run by:
```
sh train_kitti.sh
```
Before running new experiments, remember to change the model_name in train_kitti.sh and config/arguments_train_kitti.txt to be the same.

## Acknowledgements ##
We would like to acknowledge the use of code snippets from various open-source libraries and contributions from the online coding community, which have been invaluable in the development of this project. Specifically, we would like to thank the authors and maintainers of the following resources:

[CLIP](https://github.com/openai/CLIP)
[Swin Transformer](https://github.com/microsoft/Swin-Transformer)
[ExpansionNet v2](https://github.com/jchenghu/ExpansionNet_v2)
[VA-DepthNet](https://github.com/cnexah/VA-DepthNet)

## TODO: ##
Checkpoints

code refactory

updated readme