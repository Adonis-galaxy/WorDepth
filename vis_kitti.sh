model_name="0329_vis_kitti_sample"

mkdir ./models/${model_name}
CUDA_VISIBLE_DEVICES=3 python src/vis.py configs/arguments_vis_kitti.txt  2>&1 | tee ./models/${model_name}/result.log