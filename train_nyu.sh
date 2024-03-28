model_name="0327_alter_01_con_nyu_tmux3"

mkdir ./models/${model_name}
CUDA_VISIBLE_DEVICES=3 python src/train.py configs/arguments_train_nyu.txt  2>&1 | tee ./models/${model_name}/result.log