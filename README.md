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
## Run NYU-Depth-V2##
Specify GPU Number train_nyu.sh, then run by:
```
sh train_nyu.sh
```
Before running new experiments, remember to change the model_name in train_nyu.sh and config/arguments_train_nyu.txt to be the same.
## Run KITTI##
Specify GPU Number train_kitti.sh, then run by:
```
sh train_kitti.sh
```
Before running new experiments, remember to change the model_name in train_kitti.sh and config/arguments_train_kitti.txt to be the same.

