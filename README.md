# WorDepth: Variational Language Prior for Monocular Depth Estimation

Official implementation of the paper "WorDepth: Variational Language Prior for Monocular Depth Estimation"

Accepted by CVPR 2024

Paper Link: https://arxiv.org/abs/2404.03635

Presentation Video (5min): https://www.youtube.com/watch?v=QNwOFZZc8XI

Authors: Ziyao Zeng, Daniel Wang, Fengyu Yang, Hyoungseob Park, Yangchao Wu, Stefano Soatto, Byung-Woo Hong, Dong Lao, Alex Wong

## Overview ##

![teaser](figuers/teaser.png)

![pipeline](figuers/pipeline.png)

![vis_nyu](figuers/vis_nyu.png)

![poster](figuers/poster.pdf)
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

### Run NYU-Depth-V2 ###
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