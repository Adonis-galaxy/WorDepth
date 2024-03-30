model_name="0329_vis_nyu_sample"

mkdir ./models/${model_name}
CUDA_VISIBLE_DEVICES=0 python src/vis.py configs/arguments_vis_nyu.txt  2>&1 | tee ./models/${model_name}/result.log