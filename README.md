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
## Run ##
Specify GPU Number train.sh, then running by:
```
cd Stage_1

sh train.sh
```
