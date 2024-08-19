model_name="kitti_train"

mkdir ./models/${model_name}
CUDA_VISIBLE_DEVICES=1 python src/train.py configs/arguments_run_kittieigen.txt  2>&1 | tee ./models/${model_name}/result.log