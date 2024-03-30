model_name="0330_method_nyu_1_tmux1"

mkdir ./models/${model_name}
CUDA_VISIBLE_DEVICES=1 python src/train.py configs/arguments_train_nyu.txt  2>&1 | tee ./models/${model_name}/result.log