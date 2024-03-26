model_name="0322_0_wordepth_nyu_tmux2"

mkdir ./models/${model_name}
CUDA_VISIBLE_DEVICES=2 python src/train.py configs/arguments_train_nyu.txt  2>&1 | tee ./models/${model_name}/result.log