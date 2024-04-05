model_name="kitti_train"

mkdir ./models/${model_name}
CUDA_VISIBLE_DEVICES=0 python src/train.py configs/arguments_train_kittieigen.txt  2>&1 | tee ./models/${model_name}/result.log