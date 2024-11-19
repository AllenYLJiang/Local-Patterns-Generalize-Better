import random
import numpy as np
import torch
print(torch.__version__)
# from torch.utils.tensorboard import SummaryWriter
from models.STG_NF.model_flow import STG_NF
from models.training_flow import Trainer
from utils_r.data_utils import trans_list
from utils_r.optim_init import init_optimizer, init_scheduler
from args_flow import create_exp_dirs
from args_flow import init_parser, init_sub_args
# from dataset import get_dataset_and_loader
# from dataset_videoflow import VideoDatasetWithFlows
from dataset_videoflow_train import VideoDatasetWithFlows
from utils_r.train_utils import dump_args, init_model_params
from utils_r.scoring_utils import score_dataset
from utils_r.train_utils import calc_num_of_params
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d

def min_frame(scores, frame_idx):
    score_frame = []
    tmp_score = []
    frame_id = 0
    for i in range(len(scores)):
        if frame_id == frame_idx[i]:
            tmp_score.append(scores[i])
        else:
            score_frame.append(np.min(tmp_score))
            tmp_score = [scores[i]]
            frame_id += 1
    score_frame.append(np.min(tmp_score))
    return score_frame

def guass_video(scores, lengths, sigma=5):
    scores_g = np.zeros_like(scores)
    prev = 0
    for cur in lengths:
        scores_g[prev: cur] = gaussian_filter1d(scores[prev: cur], sigma)
        prev = cur
    return scores_g


def macro_auc(video, test_labels, lengths):

    prev = 0
    auc = 0
    for i, cur in enumerate(lengths):
        cur_auc = roc_auc_score(np.concatenate(([0], test_labels[prev: cur], [1])),
                             np.concatenate(([0], video[prev: cur], [sys.float_info.max])))
        auc += cur_auc
        prev = cur
    return auc / len(lengths)

    


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
    # dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained is not None))
    root = "D:\project_python\Accurate-Interpretable-VAD\data"
    flowdataset_name = "shanghaitech"
    patch_size = 112
    # flowdataset_name = "avenue" ## 做十倍降采样的训练
    all_bboxes_train = np.load(os.path.join(root, flowdataset_name, '%s_bboxes_train.npy' % flowdataset_name),
                            allow_pickle=True)
    all_bboxes_test = np.load(os.path.join(root, flowdataset_name, '%s_bboxes_test.npy' % flowdataset_name),
                              allow_pickle=True)
   
    loader_args = {'batch_size': args.batch_size, 'num_workers': 8, 'pin_memory': True}
    loader = dict()
    if pretrained:
        datasetflow_test = VideoDatasetWithFlows(dataset_name=flowdataset_name, root=root, patch_size=patch_size,
                                         train=False, sequence_length=0, all_bboxes=all_bboxes_test, normalize=True,mode = "last")
        loader["test"] = DataLoader(datasetflow_test, **loader_args, shuffle=(False))
        loader["train"] = []
        batch_flows, batch_flows_idx = datasetflow_test.__getitem__(0) # 第0个 shape
        dataset = dict()
        dataset["test"] = batch_flows.unsqueeze(0)
    else:
        datasetflow_train = VideoDatasetWithFlows(dataset_name=flowdataset_name, root=root, patch_size=patch_size,
                                            train=True, sequence_length=0, all_bboxes=all_bboxes_train, normalize=True)
        loader["train"] = DataLoader(datasetflow_train, **loader_args, shuffle=(True))
        loader["test"] = []
        batch_flows, batch_flows_idx = datasetflow_train.__getitem__(0) # 第0个 shape
        dataset = dict()
        dataset["test"] = batch_flows.unsqueeze(0)
  
    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    
    num_of_params = calc_num_of_params(model)
    trainer = Trainer(args, model, loader['train'], loader['test'],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    if pretrained:
        for i in range(11, 40):
            new_pretrained = pretrained.replace("ep0", f"ep{i}")
            # print(new_string)
            trainer.load_checkpoint(new_pretrained)
            normality_scores = trainer.test() # 144714  ---> 每帧上 每个人的score 

            idx_frame =  datasetflow_test.flows_batch_index
            
            normality_scores_m = min_frame(normality_scores, idx_frame)
            test_clip_lengths = np.load(os.path.join(root, flowdataset_name, 'test_clip_lengths.npy'))
            normality_scores_gauss = guass_video(normality_scores_m, test_clip_lengths, sigma=5)
            
            all_gt = np.load(os.path.join("D:\\project_python\\Accurate-Interpretable-VAD\\extracted_features", flowdataset_name, 'test_gt.npy'))
            auc = roc_auc_score(1-all_gt, normality_scores_gauss)

            # Logging and recording results
            print("\n-------------------------------------------------------")
            print("\033[92m Done with {}% AuC for {} samples\033[0m".format(auc * 100, len(normality_scores_m)))
            print("-------------------------------------------------------\n\n")
            np.save("save_path/"+ flowdataset_name +"/normality_scores_"+f"ep{i+1}" +".npy", normality_scores_gauss)
    else:
        # writer = SummaryWriter()
        trainer.train(log_writer=None)
        dump_args(args, args.ckpt_dir)

    

if __name__ == '__main__':
    main()
