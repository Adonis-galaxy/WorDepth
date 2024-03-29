model_name="0328_vis_kitti_alter_001_final"

mkdir ./models/${model_name}
CUDA_VISIBLE_DEVICES=0 python src/vis.py configs/arguments_vis_kitti.txt  2>&1 | tee ./models/${model_name}/result.log