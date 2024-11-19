import os
import pickle
import time
import argparse
import sys
sys.path.append("data/")

def init_args():
    parser = init_parser()
    args = parser.parse_args()
    return init_sub_args(args)


def init_sub_args(args):
    dataset = "UBnormal" if args.dataset == "UBnormal" else "ShanghaiTech"
    if args.vid_path_train and args.vid_path_test and args.pose_path_train and args.pose_path_test:
        args.vid_path = {'train': args.vid_path_train,
                         'test': args.vid_path_test}

        args.pose_path = {'train': args.pose_path_train,
                          'test': args.pose_path_test}
    else:
        args.vid_path = {'train': os.path.join(args.data_dir, dataset, 'train/images/'),
                         'test':  os.path.join(args.data_dir, dataset, 'test/frames/')}

        if args.smooth_or_predwithsmoothed_or_predwithunsmoothed == 'smooth':
            args.pose_path = {'train': os.path.join(args.data_dir, dataset, 'pose', 'test/'),
                              'test':  os.path.join(args.data_dir, dataset, 'pose', 'test/')}
        elif args.smooth_or_predwithsmoothed_or_predwithunsmoothed == 'predwithunsmoothed':
            args.pose_path = {'train': os.path.join(args.data_dir, dataset, 'train_SMM', 'train/'),
                              'test': os.path.join(args.data_dir, dataset, 'train_SMM', 'test/')}
        elif args.smooth_or_predwithsmoothed_or_predwithunsmoothed == 'predwithsmoothed_2nd_order':
            args.pose_path = {'train': os.path.join(args.data_dir, dataset, 'pose', 'train/'),
                              'test': os.path.join(args.data_dir, dataset, 'pose', 'test_second_order_refined/')}
        elif args.smooth_or_predwithsmoothed_or_predwithunsmoothed == 'predwithsmoothed':
            args.pose_path = {'train': os.path.join(args.data_dir, dataset, 'pose', 'train/'),
                              'test': os.path.join(args.data_dir, dataset, 'pose', 'test_state_machine_refined_unnormalized/')}
        elif args.smooth_or_predwithsmoothed_or_predwithunsmoothed == 'smooth_train':
            args.pose_path = {'train': os.path.join(args.data_dir, dataset, 'pose', 'test/'),
                              'test':  os.path.join(args.data_dir, dataset, 'pose', 'test/')}
        elif args.smooth_or_predwithsmoothed_or_predwithunsmoothed == 'train':
            args.pose_path = {'train': os.path.join(args.data_dir, dataset, 'train_SMM', 'train/'),
                              'test': os.path.join(args.data_dir, dataset, '', '')}

    args.pose_path["train_abnormal"] = args.pose_path_train_abnormal
    args.ckpt_dir = None
    model_args = args_rm_prefix(args, 'model_')
    return args, model_args


def init_parser(default_data_dir='data/', default_exp_dir='data/exp_dir_state/'):
    parser = argparse.ArgumentParser(prog="STG-NF")
    # General Args
    parser.add_argument('--vid_path_train', type=str, default=None, help='Path to training vids')
    parser.add_argument('--pose_path_train_abnormal', type=str, default=None, help='Path to training vids')
    parser.add_argument('--pose_path_train', type=str, default=None, help='Path to training pose')
    parser.add_argument('--vid_path_test', type=str, default=None, help='Path to test vids')
    parser.add_argument('--pose_path_test', type=str, default=None, help='Path to test pose')
    parser.add_argument('--dataset', type=str, default='ShanghaiTech',
                        choices=['ShanghaiTech', 'ShanghaiTech-HR', 'UBnormal'], help='Dataset for Eval')
    parser.add_argument('--vid_res', type=str, default=None, help='Video Res')
    parser.add_argument('--device', type=str, default='cuda:0', metavar='DEV', help='Device for feature calculation (default: \'cuda:0\')')
    parser.add_argument('--seed', type=int, metavar='S', default=999, help='Random seed, use 999 for random (default: 999)')
    parser.add_argument('--verbose', type=int, default=1, metavar='V', choices=[0, 1], help='Verbosity [1/0] (default: 1)')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, metavar='DATA_DIR', help="Path to directory holding .npy and .pkl files (default: {})".format(default_data_dir))
    parser.add_argument('--exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR', help="Path to the directory where models will be saved (default: {})".format(default_exp_dir))
    parser.add_argument('--num_workers', type=int, default=4, metavar='W', help='number of dataloader workers (0=current thread) (default: 32)')
    parser.add_argument('--plot_vid', type=int, default=0, help='Plot test videos')
    parser.add_argument('--only_test', action='store_true', help='Visualize train/test data')

    # Data Params
    parser.add_argument('--num_transform', type=int, default=2, metavar='T', help='number of transformations to use for augmentation (default: 2)')
    parser.add_argument('--headless', action='store_true', help='Remove head keypoints (14-17) and use 14 kps only. (default: False)')
    parser.add_argument('--norm_scale', '-ns', type=int, default=0, metavar='NS', choices=[0, 1], help='Scale without keeping proportions [1/0] (default: 0)')
    parser.add_argument('--prop_norm_scale', '-pns', type=int, default=1, metavar='PNS', choices=[0, 1], help='Scale keeping proportions [1/0] (default: 1)')
    parser.add_argument('--train_seg_conf_th', '-th', type=float, default=0.0, metavar='CONF_TH', help='Training set threshold Parameter (default: 0.0)')
    parser.add_argument('--seg_len', type=int, default=34, metavar='SGLEN', help='Number of frames for training segment sliding window, a multiply of 6 (default: 12)')
    parser.add_argument('--seg_stride', type=int, default=6, metavar='SGST', help='Stride for training segment sliding window')
    parser.add_argument('--specific_clip', type=int, default=None, help='Train and Eval on Specific Clip')
    parser.add_argument('--global_pose_segs', action='store_false', help='Use unormalized pose segs')

    # Mode Params
    parser.add_argument('--smooth_or_predwithsmoothed_or_predwithunsmoothed', type=str, default='', help='')

    # Model Params
    parser.add_argument('--checkpoint', type=str,
                        # default = "checkpoints/ShanghaiTech_85_9.tar", 
                        # default="/root/Downloads/STG-NF/data/exp_dir_state/ShanghaiTech/trained_on_test_set_20_pred_4/29_ep_Jul16_1747__checkpoint.pth.tar",#"/root/Downloads/STG-NF/data/exp_dir_state/ShanghaiTech/Jul13_2209/4_ep_Jul13_2211__checkpoint.pth.tar",#"/root/Downloads/STG-NF/data/exp_dir_state/ShanghaiTech/Jul10_2128_frame20_dmodel64/0_Jul10_2130__checkpoint.pth.tar",
                        # default = None,
                        # default="data\exp_dir_state\ShanghaiTech\Jul10_2128\\0_Jul10_2130__checkpoint.pth.tar",
                        # default = "D:\project_python\STG-NF\data\exp_dir_flow_train\ShanghaiTech\Jul09_2217_ped2_L3\ep0_Jul09_2217__checkpoint.pth.tar",
                        # default = "data\exp_dir_flow_train\ShanghaiTech\Jul09_1733\ep0_Jul09_1733__checkpoint.pth.tar",
                        # default = "D:\project_python\STG-NF\data\exp_dir_flow_train\ShanghaiTech\Jul09_1607_avenue\ep0_Jul09_1625__checkpoint.pth.tar",
                        # default="D:\\project_python\\STG-NF\\data\\exp_dir_flow\\ShanghaiTech\Jul08_1732\\5_checkpoint.pth.tar",
                        metavar='model', help="Path to a pretrained model")
    parser.add_argument('--batch_size', type=int, default=256,  metavar='B', help='Batch size for train')
    parser.add_argument('--epochs', '-model_e', type=int, 
                        default=30, 
                        metavar='E', help = 'Number of epochs per cycle')
    parser.add_argument('--model_optimizer', '-model_o', type=str, default='adamx', metavar='model_OPT', help = "Optimizer")
    parser.add_argument('--model_sched', '-model_s', type=str, default='exp_decay', metavar='model_SCH', help = "Optimization LR scheduler")
    parser.add_argument('--model_lr', type=float, 
                        default=5e-5,
                        # default=5e-4,
                        metavar='LR', help='Optimizer Learning Rate Parameter')
    parser.add_argument('--model_weight_decay', '-model_wd', type=float, default=5e-5, metavar='WD', help='Optimizer Weight Decay Parameter')
    parser.add_argument('--model_lr_decay', '-model_ld', type=float, default=0.99, metavar='LD', help='Optimizer Learning Rate Decay Parameter')
    parser.add_argument('--model_hidden_dim', type=int, default=0, help='Features dim dimension')
    parser.add_argument('--model_confidence', action='store_true', help='Create Figs')
    parser.add_argument('--K', type=int, default=12, help='Features dim dimension')
    parser.add_argument('--L', type=int, default=3, help='Features dim dimension')
    parser.add_argument('--R', type=float, default=3., help='Features dim dimension')
    parser.add_argument('--temporal_kernel', type=int, default=None, help='Odd integer, temporal conv size')
    # parser.add_argument('--temporal_kernel', type=int, default=3, help='Odd integer, temporal conv size')
    parser.add_argument('--edge_importance', action='store_true', help='Adjacency matrix edge weights')
    parser.add_argument('--flow_permutation', type=str, default='invcov', help='Permutation layer type')
    parser.add_argument('--adj_strategy', type=str, default='uniform', help='Adjacency matrix strategy')
    parser.add_argument('--max_hops', type=int, default=8, help='Adjacency matrix neighbours')

    return parser


def args_rm_prefix(args, prefix):
    wp_args = argparse.Namespace(**vars(args))
    args_dict = vars(args)
    wp_args_dict = vars(wp_args)
    for key, value in args_dict.items():
        if key.startswith(prefix):
            model_key = key[len(prefix):]
            wp_args_dict[model_key] = value

    return wp_args


def create_exp_dirs(experiment_dir, dirmap=''):
    time_str = time.strftime("%b%d_%H%M")

    experiment_dir = os.path.join(experiment_dir, dirmap, time_str)
    dirs = [experiment_dir]

    try:
        for dir_ in dirs:
            os.makedirs(dir_, exist_ok=True)
        print("Experiment directories created")
        return experiment_dir
    except Exception as err:
        print("Experiment directories creation Failed, error {}".format(err))
        exit(-1)


def save_dataset(dataset, fname):
    with open(fname, 'wb') as file:
        pickle.dump(dataset, file)


def load_dataset(fname):
    with open(fname, 'rb') as file:
        return pickle.load(file)

