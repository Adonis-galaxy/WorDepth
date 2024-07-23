model_name="0717_noskip_kitti_train_prob09_e200_tmux1"

mkdir ./models/${model_name}
CUDA_VISIBLE_DEVICES=1 python src/train.py configs/arguments_train_kittieigen.txt  2>&1 | tee ./models/${model_name}/result.log