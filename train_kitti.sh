model_name="0326_0_alter_0_kitti_tmux1"

mkdir ./models/${model_name}
CUDA_VISIBLE_DEVICES=1 python src/train.py configs/arguments_train_kittieigen.txt  2>&1 | tee ./models/${model_name}/result.log