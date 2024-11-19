import random
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
from models.STG_NF.model_state import ViS4mer
from models.training_state import Trainer
from utils_r.data_utils import trans_list
from utils_r.optim_init import init_optimizer, init_scheduler
from args_state import create_exp_dirs
from args_state import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils_r.train_utils import dump_args, init_model_params
from utils_r.scoring_utils import score_dataset
from utils_r.train_utils import calc_num_of_params
import os

def main():
    parser = init_parser()
    args = parser.parse_args()

    if args.smooth_or_predwithsmoothed_or_predwithunsmoothed == 'smooth':
        args.checkpoint = ""
        args.global_pose_segs = True # normalize coords
    elif args.smooth_or_predwithsmoothed_or_predwithunsmoothed == 'predwithunsmoothed':
        args.checkpoint = "data/exp_dir_state/ShanghaiTech/Nov17_1305/checkpoint.pth.tar"
        args.global_pose_segs = False # True # normalize coords
    elif args.smooth_or_predwithsmoothed_or_predwithunsmoothed == 'predwithsmoothed':
        args.checkpoint = ""
        args.global_pose_segs = True # False # normalize coords
    elif args.smooth_or_predwithsmoothed_or_predwithunsmoothed == 'predwithsmoothed_2nd_order':
        args.checkpoint = ""
        args.global_pose_segs = True # normalize coords
    elif args.smooth_or_predwithsmoothed_or_predwithunsmoothed == 'smooth_train':
        # args.checkpoint = None
        args.global_pose_segs = True  # normalize coords
    elif args.smooth_or_predwithsmoothed_or_predwithunsmoothed == 'train':
        # args.checkpoint = None
        args.global_pose_segs = False # True  # normalize coords
        args.num_workers = 0

    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)

    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)

    pretrained = vars(args).get('checkpoint', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained is not None))

    # model_args = init_model_params(args, dataset)
    # model = STG_NF(**model_args)
    input_frame = 33  # dmodel 
    model = ViS4mer(d_input=768, l_max=input_frame, d_output=(34 - input_frame)*768, d_model=64, n_layers=3) # 18 x 2

    trainer = Trainer(args, model, loader['train'], loader['test'],input_frame,
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))

    # preprocess to smooth pose sequences with state machines
    # note that the weights loaded here should be the same as later
    if args.smooth_or_predwithsmoothed_or_predwithunsmoothed == 'smooth':
        conf_gain_thresh = 0.08 # real-time decide, if the predicted pose has a confidence of over 8% higher than detected pose, replace it
        trainer.load_checkpoint(pretrained)
        for input_file in [x for x in os.listdir('data/ShanghaiTech/pose/test') if '_tracked_person' in x]:#[105:]:
            print(input_file)
            trainer.smooth_pose_sequences_with_state_machine(os.path.join('data/ShanghaiTech/pose/test', input_file), \
                                                             os.path.join('data/ShanghaiTech/pose/test_state_machine_refined_unnormalized', input_file), 34, input_frame, conf_gain_thresh)
        return

    if pretrained:
        for i in range(3, 4):#30):
            new_pretrained = pretrained.replace("0_Jul10", f"{i}_Jul10")
            trainer.load_checkpoint(new_pretrained)
            normality_scores = trainer.test() # 144714  ---> 
            auc, scores = score_dataset(normality_scores, dataset["test"].metadata, args.pose_path, args=args)
            
            np.save("save_path/S4shanghai_d64_f20/normality_scores_"+f"ep{i+1}" +".npy", scores)
            # Logging and recording results
            print("\n-------------------------------------------------------")
            print("\033[92m Done with {}% AuC for {} samples\033[0m".format(auc * 100, scores.shape[0]))
            print("-------------------------------------------------------\n\n")
    else:
        # writer = SummaryWriter()
        trainer.train(log_writer=None)
        dump_args(args, args.ckpt_dir)

    # auc, scores = score_dataset(normality_scores, dataset["test"].metadata, args=args)



if __name__ == '__main__':
    main()
