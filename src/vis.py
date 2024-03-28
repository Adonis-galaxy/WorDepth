import torch
import os, sys, time
import argparse
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import compute_errors, eval_metrics, block_print, enable_print, convert_arg_line_to_args
from networks.wordepth import WorDepth
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import shutil
import os

parser = argparse.ArgumentParser(description='WorDepth PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='WorDepth')
parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)

# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--prior_mean',                type=float, help='prior mean of depth', default=1.54)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=5000)

# Training
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)

# Online eval
parser.add_argument('--eval_before_train',         action='store_true')
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')
# WorDepth
parser.add_argument('--weight_kld',            type=float, default=1e-3)
parser.add_argument('--alter_prob',            type=float, default=0.1)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader





def main():
    args_out_path = os.path.join(args.log_directory, args.model_name)
    os.makedirs(args_out_path, exist_ok=True)
    vis_pred_path = os.path.join(args_out_path, "vis_pred")
    vis_sample_path = os.path.join(args_out_path, "vis_sample_from_gaussian")
    os.makedirs(vis_pred_path, exist_ok=True)
    os.makedirs(vis_sample_path, exist_ok=True)

    depth_gt_save_path = os.path.join(args_out_path, "depth_gt")
    image_save_path = os.path.join(args_out_path, "image")
    os.makedirs(depth_gt_save_path, exist_ok=True)
    os.makedirs(image_save_path, exist_ok=True)

    txt_save_path = os.path.join(args_out_path, "txt")
    os.makedirs(txt_save_path, exist_ok=True)

    model = WorDepth(pretrained=args.pretrain,
                       max_depth=args.max_depth,
                       prior_mean=args.prior_mean,
                       img_size=(args.input_height, args.input_width),
                       weight_kld=args.weight_kld,
                       alter_prob=args.alter_prob)

    model = torch.nn.DataParallel(model)
    model.cuda()

    print("== Model Initialized")
    # ===== Load CKPT =====
    print("== Loading checkpoint '{}'".format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    dataloader_eval = NewDataLoader(args, 'online_eval')

    # ===== Vis ======
    model.eval()
    post_process=True

    eval_measures = torch.zeros(10).cuda()
    for idx, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = eval_sample_batched['image'].cuda(non_blocking=True)
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue

            # Read Text and Image Feature
            text_feature_list = []
            image_path = args.data_path_eval+eval_sample_batched['sample_path'][0].split(' ')[0]
            for i in range(len(eval_sample_batched['sample_path'])):

                # TODO: This is data loading and should be in the dataloader so we can
                # make use of multithreading
                if args.dataset == "nyu":
                    text_feature_path = args.data_path_eval+eval_sample_batched['sample_path'][i].split(' ')[0][:-4]+'.pt'
                elif args.dataset == "kitti":
                    text_feature_path = args.data_path_eval + \
                        eval_sample_batched['sample_path'][i].split(' ')[0][:-4]+'_txt_feat.pt'

                text_feature = torch.load(text_feature_path, map_location=image.device)
                text_feature_list.append(text_feature)

            text_feature_list = torch.cat(text_feature_list, dim=0)

            # Forwarding Model
            pred_depth = model(image, text_feature_list, sample_from_gaussian_eval=False)
            sample_depth = model(image, text_feature_list, sample_from_gaussian_eval=True)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            sample_depth = sample_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        sample_depth[sample_depth < args.min_depth_eval] = args.min_depth_eval
        sample_depth[sample_depth > args.max_depth_eval] = args.max_depth_eval
        sample_depth[np.isinf(sample_depth)] = args.max_depth_eval
        sample_depth[np.isnan(sample_depth)] = args.min_depth_eval

        # Vis
        ori_image = Image.open(image_path)
        if args.dataset == 'nyu':
            vis_depth = pred_depth[45:472, 43:608]
            vis_sample_depth = sample_depth[45:472, 43:608]
            vis_image = ori_image.crop((43, 45, 608, 472))
            vis_gt_depth = gt_depth[45:472, 43:608]
        else:
            vis_depth = pred_depth
            vis_sample_depth = sample_depth
            vis_image = ori_image
            vis_gt_depth = gt_depth

        height, width = vis_image.height, vis_image.width

        # Set the dpi of the figure to match the image resolution
        dpi = plt.rcParams['figure.dpi']

        # Calculate the figsize to match the image resolution
        figsize = width / dpi, height / dpi

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(vis_depth, cmap='viridis')  # Assuming depth map is a 2D numpy array
        ax.axis('off')  # Disable axis
        fig_path = os.path.join(vis_pred_path, f'pred_depth_{idx + 1}.png')
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)  # Save the figure without extra padding

        fig_path = os.path.join(image_save_path, f'image_{idx + 1}.png')
        vis_image.save(fig_path)

        ax.imshow(vis_gt_depth, cmap='viridis')  # Assuming depth map is a 2D numpy array
        ax.axis('off')  # Disable axis
        fig_path = os.path.join(depth_gt_save_path, f'depth_gt_{idx + 1}.png')
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)  # Save the figure without extra padding

        ax.imshow(vis_sample_depth, cmap='viridis')  # Assuming depth map is a 2D numpy array
        ax.axis('off')  # Disable axis
        fig_path = os.path.join(vis_sample_path, f'sample_depth_{idx + 1}.png')
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)  # Save the figure without extra padding

        plt.close(fig)

        # Move the txt to the destination directory
        txt_path = os.path.join(txt_save_path, f'caption_{idx + 1}.txt')
        source_path = args.data_path_eval+eval_sample_batched['sample_path'][i].split(' ')[0][:-4]+'.txt'
        shutil.move(source_path, txt_path)

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda()
        eval_measures[9] += 1

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                    'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                    'd3'))
    for i in range(8):
        print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.4f}'.format(eval_measures_cpu[8]))

if __name__ == '__main__':
    main()
