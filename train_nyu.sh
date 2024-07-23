model_name="0717_noskip_nyu_train_prob09_e200_tmux0"

mkdir ./models/${model_name}
CUDA_VISIBLE_DEVICES=0 python src/train.py configs/arguments_train_nyu.txt  2>&1 | tee ./models/${model_name}/result.log