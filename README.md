## Coming Soon...... ##
Checkpoints, code refactory, updated readme

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

For simplicity, we have extracted image caption using [ExpansionNet v2](https://github.com/jchenghu/ExpansionNet_v2) in "./text" , and extracted their CLIP text features in "./text_feat". While running, the model will automatically load text features.
## Run NYU-Depth-V2 ##
Specify GPU Number train_nyu.sh, then run by:
```
sh train_nyu.sh
```
Before running new experiments, remember to change the model_name in train_nyu.sh and config/arguments_train_nyu.txt to be the same.
## Run KITTI ##
Specify GPU Number train_kitti.sh, then run by:
```
sh train_kitti.sh
```
Before running new experiments, remember to change the model_name in train_kitti.sh and config/arguments_train_kitti.txt to be the same.
