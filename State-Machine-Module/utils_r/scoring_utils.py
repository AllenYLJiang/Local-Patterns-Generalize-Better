import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataset import shanghaitech_hr_skip
import joblib

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def score_dataset(score, metadata, input_folder_name, args=None):
    gt_arr, scores_arr, test_video_names = get_dataset_scores(score, metadata, input_folder_name['test'], args=args)
    scores_arr = smooth_scores(scores_arr)
    gt_np = np.concatenate(gt_arr)  # 40791帧
    scores_np = np.concatenate(scores_arr) # 40791
    auc = score_auc(scores_np, gt_np)
    # video-level auc
    video_level_auc = []
    for video_idx in range(51):
        video_level_auc.append(score_auc(scores_arr[video_idx], gt_arr[video_idx]))
    video_level_auc.append(None)
    for video_idx in range(52, len(gt_arr)):
        video_level_auc.append(score_auc(scores_arr[video_idx], gt_arr[video_idx]))
    joblib.dump(dict(zip([x.split('.')[0] for x in test_video_names], [scores_arr[key] for key in range(len(test_video_names))])), input_folder_name['test'][:-1] + '_scores.pkl')
    joblib.dump(dict(zip([x.split('.')[0] for x in test_video_names], [gt_arr[key] for key in range(len(test_video_names))])), input_folder_name['test'][:-1] + '_gt.pkl')
    return auc, scores_np


def get_dataset_scores(scores, metadata, input_test_dir, args=None):
    dataset_gt_arr = []
    dataset_scores_arr = []
    metadata_np = np.array(metadata)

    if args.dataset == 'UBnormal':
        pose_segs_root = 'data/UBnormal/pose/test'
        clip_list = os.listdir(pose_segs_root)
        clip_list = sorted(
            fn.replace("alphapose_tracked_person.json", "tracks.txt") for fn in clip_list if fn.endswith('.json'))
        per_frame_scores_root = 'data/UBnormal/gt/'
    else:
        per_frame_scores_root = 'data/ShanghaiTech/gt/test_frame_mask/'
        clip_list = [x for x in os.listdir(per_frame_scores_root) if (x.split('.')[0] + '_alphapose_tracked_person.json') in os.listdir(input_test_dir)]
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))

    print("Scoring {} clips".format(len(clip_list)))
    clip_list_result = []
    for clip in tqdm(clip_list):
        clip_gt, clip_score = get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args)
        if clip_score is not None:
            dataset_gt_arr.append(clip_gt)
            dataset_scores_arr.append(clip_score)
            clip_list_result.append(clip)

    scores_np = np.concatenate(dataset_scores_arr, axis=0)
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    index = 0
    for score in range(len(dataset_scores_arr)):
        for t in range(dataset_scores_arr[score].shape[0]):
            dataset_scores_arr[score][t] = scores_np[index]
            index += 1

    return dataset_gt_arr, dataset_scores_arr, clip_list_result


def score_auc(scores_np, gt):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    ############## 魔改 加点 vecolity信息 ###### 0 1 是反的
    # # np.save('utils\\scores_nfpose.npy', scores_np)
    # vel = np.load('utils\\final_scores_velocity.npy')
    # final_score =  scores_np - vel

    # auc = roc_auc_score(gt, final_score)
    auc = roc_auc_score(gt, scores_np)

    return auc


def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr


def get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args):
    if args.dataset == 'UBnormal':
        type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_tracks.*', clip)[0]
        clip_id = type + "_" + clip_id
    else:
        scene_id, clip_id = [int(i) for i in clip.replace("label", "001").split('.')[0].split('_')]
        if shanghaitech_hr_skip((args.dataset == 'ShanghaiTech-HR'), scene_id, clip_id):
            return None, None
    clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                  (metadata_np[:, 0] == scene_id))[0]   ## metadata 记录了属于人物属于哪一帧的索引
    clip_metadata = metadata[clip_metadata_inds] # 1053 4
    clip_fig_idxs = set([arr[2] for arr in clip_metadata])
    clip_res_fn = os.path.join(per_frame_scores_root, clip)
    clip_gt = np.load(clip_res_fn)
    if args.dataset != "UBnormal":
        clip_gt = np.ones(clip_gt.shape) - clip_gt  # 1 is normal, 0 is abnormal
    scores_zeros = np.ones(clip_gt.shape[0]) * np.inf
    if len(clip_fig_idxs) == 0:
        clip_person_scores_dict = {0: np.copy(scores_zeros)}
    else:
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

    for person_id in clip_fig_idxs:
        person_metadata_inds = \
            np.where( # 根据人物的图形索引从元数据中获取人物的元数据索引
                (metadata_np[:, 1] == clip_id) & (metadata_np[:, 0] == scene_id) & (metadata_np[:, 2] == person_id))[0]
        pid_scores = scores[person_metadata_inds] # 获取人物的得分

        pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds]).astype(int) # 获取人物在剪辑中的帧索引
        clip_person_scores_dict[person_id][pid_frame_inds + int(args.seg_len / 2)] = pid_scores

    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))  # 5 个人， 其实就是取每一帧中的得分最低的人
    clip_score = np.amin(clip_ppl_score_arr, axis=0)  # 最小的值 0为异常 1为正常 

    return clip_gt, clip_score
