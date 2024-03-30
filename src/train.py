import torch
import os, sys, time
import argparse
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import compute_errors, eval_metrics, block_print, enable_print, convert_arg_line_to_args
from networks.wordepth import WorDepth

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
parser.add_argument('--store_freq',            type=int, default=0)


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader


def online_eval(model, dataloader_eval, post_process=False):
    eval_measures = torch.zeros(10).cuda()
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = eval_sample_batched['image'].cuda(non_blocking=True)
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue

            # Read Text and Image Feature
            text_feature_list = []
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
            pred_depth = model(image, text_feature_list, sample_from_gaussian=False)

            pred_depth = pred_depth.cpu().numpy().squeeze()
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
    return eval_measures_cpu



def main_worker(args):
    model = WorDepth(pretrained=args.pretrain,
                       max_depth=args.max_depth,
                       prior_mean=args.prior_mean,
                       img_size=(args.input_height, args.input_width),
                       weight_kld=args.weight_kld,
                       alter_prob=args.alter_prob)
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    model = torch.nn.DataParallel(model)
    model.cuda()

    print("== Model Initialized")

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    # Training parameters
    optimizer = torch.optim.Adam([{'params': model.module.parameters()}],
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)

    model_just_loaded = False
    # ===== Load CKPT =====
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if not args.retrain:
                try:
                    global_step = checkpoint['global_step']
                    best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                    best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                    best_eval_steps = checkpoint['best_eval_steps']
                except KeyError:
                    print("Could not load values for online evaluation")

            print("== Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True
        del checkpoint

    torch.backends.cudnn.benchmark = True

    dataloader = NewDataLoader(args, 'train')
    dataloader_eval = NewDataLoader(args, 'online_eval')

    # ===== Evaluation before training ======
    if args.eval_before_train is True:
        model.eval()
        with torch.no_grad():
            eval_measures = online_eval(model, dataloader_eval, post_process=True)
        model.train()
    # Logging
    if args.do_online_eval:
        if args.eval_summary_directory != '':
            eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
        else:
            eval_summary_path = os.path.join(args.log_directory, args.model_name, 'summary')
        summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else args.learning_rate

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    while epoch < args.num_epochs:

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            image = sample_batched['image'].cuda(non_blocking=True)
            depth_gt = sample_batched['depth'].cuda(non_blocking=True)

            # Read Text and Image Feature
            text_feature_list = []
            for i in range(len(sample_batched['sample_path'])):
                if args.dataset == "nyu":
                    text_feature_path = args.data_path+sample_batched['sample_path'][i].split(' ')[0][:-4]+'.pt'
                elif args.dataset == "kitti":
                    text_feature_path = args.data_path + \
                        sample_batched['sample_path'][i].split(' ')[0][:-4]+'_txt_feat.pt'

                text_feature = torch.load(text_feature_path, map_location=image.device)
                text_feature_list.append(text_feature)

            text_feature_list = torch.cat(text_feature_list, dim=0)
            depth_est, loss = model(image, text_feature_list, depth_gt)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                summary_writer.add_scalar("training_loss", loss.item(), int(global_step))
                print('epoch:', epoch, 'global_step:', global_step, 'loss:', loss.item(), flush=True)

            if args.store_freq != 0 and global_step % args.store_freq == 0:
                checkpoint = {'global_step': global_step,
                             'model': model.state_dict()}
                model_save_name = '/model-{}'.format(global_step)
                torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                model.eval()
                with torch.no_grad():
                    eval_measures = online_eval(model, dataloader_eval, post_process=False)
                if eval_measures is not None:
                    for i in range(9):
                        summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': model.state_dict(),
                                        #   'optimizer': optimizer.state_dict(),
                                          'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                          'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                          'best_eval_steps': best_eval_steps
                                          }
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                    summary_writer.flush()
                model.train()
                block_print()
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1



def main():
    if args.mode != 'train':
        print('train.py is only for training.')
        return -1

    args_out_path = os.path.join(args.log_directory, args.model_name)
    os.makedirs(args_out_path, exist_ok=True)
    os.system('cp ' + sys.argv[1] + ' ' + args_out_path)
    os.system('cp ' + "src/train.py" + ' ' + args_out_path + "/train.py.backup")
    os.system('cp ' + "src/networks/wordepth.py" + ' ' + args_out_path + "/wordepth.py.backup")

    torch.cuda.empty_cache()

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    main_worker(args)


if __name__ == '__main__':
    main()
