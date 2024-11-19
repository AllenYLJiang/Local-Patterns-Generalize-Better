import random
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import create_exp_dirs
from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params
from utils.scoring_utils import score_dataset
from utils.train_utils import calc_num_of_params
from torch.utils.data import DataLoader


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
    # ### loarder 改一下 ########################################################
    # pretrained = vars(args).get('checkpoint', None)
    # import os
    # loader_flows = {}
    # root="D:\\project_python\\Accurate-Interpretable-VAD\\data"
    # from flow_video_dataset import VideoDatasetWithFlows
    # args.dataset_name = "shanghaitech"
    # all_bboxes_train = np.load(os.path.join(root, args.dataset_name, '%s_bboxes_train.npy' % args.dataset_name),
    #                            allow_pickle=True)
    # all_bboxes_test = np.load(os.path.join(root, args.dataset_name, '%s_bboxes_test.npy' % args.dataset_name),
    #                           allow_pickle=True)
    # test_dataset = VideoDatasetWithFlows(dataset_name=args.dataset_name, root=root,
    #                                      train=False, sequence_length=0, all_bboxes=all_bboxes_test, normalize=False, mode='last')
    # train_dataset = VideoDatasetWithFlows(dataset_name=args.dataset_name, root=root,
    #                                       train=True, sequence_length=0, all_bboxes=all_bboxes_train, normalize=True)
    
    # # split = 'test'
    # loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    # loader_flows['test'] = DataLoader(test_dataset, **loader_args, shuffle=(False))
    # loader_flows['train'] = DataLoader(train_dataset, **loader_args, shuffle=(True))
    
    # model_args = init_model_params(args, train_dataset)
    # model = STG_NF(**model_args)
    # num_of_params = calc_num_of_params(model)
    
    # trainer = Trainer(args, model, loader_flows['train'], loader_flows['test'],
    #                   optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
    #                   scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    # if pretrained:
    #     trainer.load_checkpoint(pretrained)
    # else:
    #     # writer = SummaryWriter()
    #     trainer.train(log_writer=None)
    #     dump_args(args, args.ckpt_dir)
    #############################################################################################
    #########################################################################################
    pretrained = vars(args).get('checkpoint', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained is not None))

    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    num_of_params = calc_num_of_params(model)
    

    trainer = Trainer(args, model, loader['train'], loader['test'],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    if pretrained:
        trainer.load_checkpoint(pretrained)
    else:
        # writer = SummaryWriter()
        trainer.train(log_writer=None)
        dump_args(args, args.ckpt_dir)

    normality_scores = trainer.test() # 144714  ---> 每帧上 每个人的score 
    auc, scores = score_dataset(normality_scores, dataset["test"].metadata, args=args)

    # Logging and recording results
    print("\n-------------------------------------------------------")
    print("\033[92m Done with {}% AuC for {} samples\033[0m".format(auc * 100, scores.shape[0]))
    print("-------------------------------------------------------\n\n")


if __name__ == '__main__':
    main()
