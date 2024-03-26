exp_name="0322_0_wordepth_1en2_eps_with_mean_kitti_tmux2"

mkdir ./models/${exp_name}
CUDA_VISIBLE_DEVICES=2 python src/train.py configs/arguments_train_kittieigen.txt  2>&1 | tee ./models/${exp_name}/result.log/media/home/zyzeng/wordepth_backup/Stage_1