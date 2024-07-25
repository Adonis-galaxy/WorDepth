model_name="0723_noskip_nyu_train_prob05_e200_tmux2"

mkdir ./models/${model_name}
CUDA_VISIBLE_DEVICES=2 python src/train.py configs/arguments_train_nyu.txt  2>&1 | tee ./models/${model_name}/result.log