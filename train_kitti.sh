model_name="0329_final_ckpt_eval_con"

mkdir ./models/${model_name}
CUDA_VISIBLE_DEVICES=0 python src/train.py configs/arguments_train_kittieigen.txt  2>&1 | tee ./models/${model_name}/result.log