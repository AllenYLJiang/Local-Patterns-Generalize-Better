import random
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
# from models.STG_NF.model_pose import STG_NF
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


def main():
    parser = init_parser()
    args = parser.parse_args()

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
    ### 20 帧 预测 4帧
    input_frame = 20  # dmodel 改大
    model = ViS4mer(d_input=36, l_max=input_frame, d_output=(24 - input_frame)*36, d_model=64, n_layers=3) # 18 x 2

    trainer = Trainer(args, model, loader['train'], loader['test'],input_frame,
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    if pretrained:
        for i in range(3, 30):
            new_pretrained = pretrained.replace("0_Jul10", f"{i}_Jul10")
            trainer.load_checkpoint(new_pretrained)
            normality_scores = trainer.test() # 144714  ---> 每帧上 每个人的score 
            auc, scores = score_dataset(normality_scores, dataset["test"].metadata, args=args)
            
            np.save("save_path/S4shanghai_d64_f20/normality_scores_"+f"ep{i+1}" +".npy", scores)
            # Logging and recording results
            print("\n-------------------------------------------------------")
            print("\033[92m Done with {}% AuC for {} samples\033[0m".format(auc * 100, scores.shape[0]))
            print("-------------------------------------------------------\n\n")
    else:
        # writer = SummaryWriter()
        trainer.train(log_writer=None)
        dump_args(args, args.ckpt_dir)

    # normality_scores = trainer.test() # 144714  ---> 每帧上 每个人的score 
    # auc, scores = score_dataset(normality_scores, dataset["test"].metadata, args=args)



if __name__ == '__main__':
    main()
