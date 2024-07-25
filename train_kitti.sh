model_name="0725_noskip_kitti_train_prob05_e200_tmux3"

mkdir ./models/${model_name}
CUDA_VISIBLE_DEVICES=3 python src/train.py configs/arguments_train_kittieigen.txt  2>&1 | tee ./models/${model_name}/result.log