"""
The code in this file is adapted from:
https://github.com/LiUzHiAn/hf2vad/blob/master/datasets/dataset.py
"""

import torch
import numpy as np
import cv2
from collections import OrderedDict
import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import scipy.io as scio
from tqdm import tqdm
def img_tensor2numpy(img):
    # mutual transformation between ndarray-like imgs and Tensor-like images
    # both intensity and rgb images are represented by 3-dim data
    if isinstance(img, np.ndarray):
        return torch.from_numpy(np.transpose(img, [2, 0, 1]))
    else:
        return np.transpose(img, [1, 2, 0]).numpy()


def img_batch_tensor2numpy(img_batch):
    # both intensity and rgb image batch are represented by 4-dim data
    if isinstance(img_batch, np.ndarray):
        if len(img_batch.shape) == 4:
            return torch.from_numpy(np.transpose(img_batch, [0, 3, 1, 2]))
        else:
            return torch.from_numpy(np.transpose(img_batch, [0, 1, 4, 2, 3]))
    else:
        if len(img_batch.numpy().shape) == 4:
            return np.transpose(img_batch, [0, 2, 3, 1]).numpy()
        else:
            return np.transpose(img_batch, [0, 1, 3, 4, 2]).numpy()

def get_foreground(img, bboxes, patch_size):
    """
    Cropping the object area according to the bouding box, and resize to patch_size
    :param img: [#frame,c,h,w]
    :param bboxes: [#,4]
    :param patch_size: 32
    :return:
    """
    img_patches = []
    if len(img.shape) == 3:
        for i in range(len(bboxes)):
            x_min, x_max = np.int32(np.ceil(bboxes[i][0])), np.int32(np.ceil(bboxes[i][2]))
            y_min, y_max = np.int32(np.ceil(bboxes[i][1])), np.int32(np.ceil(bboxes[i][3]))
            cur_patch = img[:, y_min:y_max, x_min:x_max]
            cur_patch = cv2.resize(np.transpose(cur_patch, [1, 2, 0]), (patch_size, patch_size))
            img_patches.append(np.transpose(cur_patch, [2, 0, 1]))
        img_patches = np.array(img_patches)
    elif len(img.shape) == 4:
        for i in range(len(bboxes)):
            x_min, x_max = np.int32(np.ceil(bboxes[i][0])), np.int32(np.ceil(bboxes[i][2]))
            y_min, y_max = np.int32(np.ceil(bboxes[i][1])), np.int32(np.ceil(bboxes[i][3]))
            cur_patch_set = img[:, :, y_min:y_max, x_min:x_max]
            tmp_set = []
            for j in range(img.shape[0]):  # temporal patches
                cur_patch = cur_patch_set[j]
                cur_patch = cv2.resize(np.transpose(cur_patch, [1, 2, 0]),
                                       (patch_size, patch_size))
                tmp_set.append(np.transpose(cur_patch, [2, 0, 1]))
            cur_cube = np.array(tmp_set)  # spatial-temporal cube for each bbox
            img_patches.append(cur_cube)  # all spatial-temporal cubes in a single frame
        img_patches = np.array(img_patches)
    return img_patches  # [num_bboxes, frames_num, C, patch_size, patch_size]

def get_foreground_8(img, bboxes, patch_size):
    """
    Cropping the object area according to the bouding box, and resize to patch_size
    :param img: [#frame,c,h,w]
    :param bboxes: [#,4]
    :param patch_size: 32
    :return:
    """
    img_patches = []
    if len(img.shape) == 3:
        for i in range(len(bboxes)):
            x_min, x_max = np.int32(np.ceil(bboxes[i][0])), np.int32(np.ceil(bboxes[i][2]))
            y_min, y_max = np.int32(np.ceil(bboxes[i][1])), np.int32(np.ceil(bboxes[i][3]))
            cur_patch = img[:, y_min:y_max, x_min:x_max]
            cur_patch = cv2.resize(np.transpose(cur_patch, [1, 2, 0]), (patch_size, patch_size))
            img_patches.append(np.transpose(cur_patch, [2, 0, 1]))
        img_patches = np.array(img_patches)
    elif len(img.shape) == 4:
        for i in range(len(bboxes)):
            x_min, x_max = np.int32(np.ceil(bboxes[i][0])), np.int32(np.ceil(bboxes[i][2]))
            y_min, y_max = np.int32(np.ceil(bboxes[i][1])), np.int32(np.ceil(bboxes[i][3]))
            cur_patch_set = img[:, :, y_min:y_max, x_min:x_max]
            tmp_set = []
            for j in range(img.shape[0]):  # temporal patches
                cur_patch = cur_patch_set[j]
                cur_patch = cv2.resize(np.transpose(cur_patch, [1, 2, 0]),
                                       (patch_size, patch_size))
                tmp_set.append(np.transpose(cur_patch, [2, 0, 1]))
            cur_cube = np.array(tmp_set)  # spatial-temporal cube for each bbox
            img_patches.append(cur_cube)  # all spatial-temporal cubes in a single frame
        img_patches = np.array(img_patches)
    return img_patches  # [num_bboxes, frames_num, C, patch_size, patch_size]

### get_track
def get_foreground_track(img, bboxes, patch_size): # img 一帧图像
    """
    Cropping the object area according to the bouding box, and resize to patch_size
    :param img: [#frame,c,h,w]
    :param bboxes: [#,4]
    :param patch_size: 32
    :return:
    """
    img_patches = []
    object_id = []
    if len(img.shape) == 3:
        for i in range(len(bboxes)):
            x_min, x_max = np.int32(np.ceil(bboxes[i][1])), np.int32(np.ceil(bboxes[i][3]+bboxes[i][1]))
            y_min, y_max = np.int32(np.ceil(bboxes[i][2])), np.int32(np.ceil(bboxes[i][4]+bboxes[i][2]))
            cur_patch = img[:, y_min:y_max, x_min:x_max]
            cur_patch = cv2.resize(np.transpose(cur_patch, [1, 2, 0]), (patch_size, patch_size))
            img_patches.append(np.transpose(cur_patch, [2, 0, 1]))
        img_patches = np.array(img_patches)
    elif len(img.shape) == 4:
        for i in range(len(bboxes)):
            x_min, x_max = np.int32(np.ceil(bboxes[i][1])), np.int32(np.ceil(bboxes[i][3]+bboxes[i][1]))
            y_min, y_max = np.int32(np.ceil(bboxes[i][2])), np.int32(np.ceil(bboxes[i][4]+bboxes[i][2]))
            x_min,y_min = max(0,x_min),max(0,y_min)
            cur_patch_set = img[:, :, y_min:y_max, x_min:x_max]
            tmp_set = []
            for j in range(img.shape[0]):  # temporal patches
                cur_patch = cur_patch_set[j]
                cur_patch = cv2.resize(np.transpose(cur_patch, [1, 2, 0]),
                                       (patch_size, patch_size))
                tmp_set.append(np.transpose(cur_patch, [2, 0, 1]))
            cur_cube = np.array(tmp_set)  # spatial-temporal cube for each bbox
            img_patches.append(cur_cube)  # all spatial-temporal cubes in a single frame
            object_id.append(np.int32(np.ceil(bboxes[i][0])))
        img_patches = np.array(img_patches)
        object_id = np.array(object_id)
    return img_patches,object_id   # [num_bboxes, frames_num, C, patch_size, patch_size]

def sort_file_names(frame_name):
    # ## debug
    # print(frame_name)
    # print(frame_name.split('\\')[-1].split('.')[0])
    frame_name = int(frame_name.split('\\')[-1].split('.')[0])
    return frame_name

class VideoDatasetWithFlows(Dataset):
    def __init__(self, dataset_name, root, train=True, sequence_length=0, mode="last",
                 all_bboxes=None, patch_size=224, normalize=False, bboxes_extractions=False):
        super(VideoDatasetWithFlows, self).__init__()
        self.dataset_name = dataset_name
        self.root = os.path.join(root, dataset_name)
        self.train = train
        self.videos = OrderedDict()
        self.flows = OrderedDict()
        self.frame_addresses = []
        self.frame_addresses_flows = []
        self.frame_video_idx = []
        self.sequence_length = sequence_length
        self.mode = mode
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size
        self.normalize = normalize
        self.bboxes_extractions = bboxes_extractions
        self.__initialize()
        self.num_of_frames = len(self.frame_addresses)


    def __len__(self):
        # return self.num_of_frames
        return len(self.flows_batch_all)

    def __initialize(self):
        if self.train:
            self.root_frames = os.path.join(self.root, 'training', 'frames/*')
            if self.dataset_name == 'shanghaitech':
                self.root_frames_flows = os.path.join("G:\ck_flownet",self.dataset_name, 'training', 'flows_1/*')
                # self.root_frames_flows = os.path.join("/media/allen/Seagate_11/ck_flownet",self.dataset_name, 'training', 'flows/*')
                videos_1 = glob.glob(self.root_frames_flows)
                videos_2 = glob.glob("H:\\flows_2\\*")
                videos = videos_1 + videos_2
            else:
                self.root_frames_flows = os.path.join(self.root, 'training', 'flows/*')
                videos = glob.glob(self.root_frames_flows)
                # ########################################
        else:
            self.root_frames = os.path.join(self.root, 'testing', 'frames/*')
            
            if self.dataset_name == 'shanghaitech':
                # self.root_frames_flows = os.path.join("G:\ck_flownet",self.dataset_name, 'testing', 'flows_1/*')
                self.root_frames_flows = os.path.join("G:\ck_flownet",self.dataset_name, 'testing', 'flows/*')
            else:
                self.root_frames_flows = os.path.join(self.root, 'testing', 'flows/*')
            videos = glob.glob(self.root_frames_flows)
        for i, video_path in enumerate(sorted(videos)):
            self.flows[i] = {}
            self.flows[i]['path'] = video_path
            self.flows[i]['frames'] = glob.glob(os.path.join(video_path, '*.npy'))
            self.flows[i]['frames'].sort(key=sort_file_names)
            self.frame_addresses_flows += self.flows[i]['frames']
            self.flows[i]['length'] = len(self.flows[i]['frames'])

        videos = glob.glob(self.root_frames)
        for i, video_path in enumerate(sorted(videos)):
            self.videos[i] = {}
            self.videos[i]['path'] = video_path
            self.videos[i]['frames'] = glob.glob(os.path.join(video_path, '*.jpg'))
            self.videos[i]['frames'].sort(key=sort_file_names)
            self.frame_addresses += self.videos[i]['frames']
            self.videos[i]['length'] = len(self.videos[i]['frames'])
            self.frame_video_idx += [i] * self.videos[i]['length']

        if not self.train:
            self.get_gt()

        # gendata
        self.data_all = self.gendata()

    def get_gt(self):
        if self.dataset_name == 'ped2':
            mat_name = self.dataset_name + '.mat'
            mat_path = os.path.join(self.root, mat_name)
            abnormal_mat = scio.loadmat(mat_path, squeeze_me=True)['gt']
            self.all_gt = []
            for i, (_, video) in enumerate(self.videos.items()):
                length = video['length']
                sub_video_gt = np.zeros((length,), dtype=np.int8)
                one_abnormal = abnormal_mat[i]
                if one_abnormal.ndim == 1:
                    one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))
                for j in range(one_abnormal.shape[1]):
                    start = one_abnormal[0, j] - 1
                    end = one_abnormal[1, j]
                    sub_video_gt[start: end] = 1
                sub_video_gt = torch.from_numpy(sub_video_gt)
                self.all_gt.append(sub_video_gt)
            self.all_gt = torch.cat(self.all_gt, 0)

        elif self.dataset_name == 'avenue':
            ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
                   '18', '19', '20', '21']
            mat_name = self.dataset_name + '.mat'
            mat_path = os.path.join(self.root, mat_name)
            abnormal_mat = scio.loadmat(mat_path)
            self.all_gt = []
            for id in ids:
                self.all_gt.append(abnormal_mat[id][0])
            self.all_gt = np.concatenate(self.all_gt, axis=0)
            self.all_gt = torch.from_numpy(self.all_gt)

        elif self.dataset_name == 'shanghaitech':
            frame_mask = os.path.join(self.root, 'testing/test_frame_mask/*')
            frame_mask = glob.glob(frame_mask)
            frame_mask.sort()
            self.all_gt = []
            for i, (_, video) in enumerate(self.videos.items()):
                gt = torch.from_numpy(np.load(frame_mask[i]))
                self.all_gt.append(gt)
            self.all_gt = torch.cat(self.all_gt, 0)

    def get_sequence(self, index):
        """
        This function gets an index and returns a clip (of size self.sequence_length).
        """
        if index - self.sequence_length < 0:
            start_idx = 0
        else:
            start_idx = index - self.sequence_length

        if self.mode == "last":
            end_idx = index
            clip_length = self.sequence_length + 1  # future frame prediction
        else:   # Means that self.mode == "middle"
            if index + self.sequence_length > self.num_of_frames - 1:
                end_idx = self.num_of_frames - 1
            else:
                end_idx = index + self.sequence_length
            clip_length = 2 * self.sequence_length + 1

        main_frame_video_idx = self.frame_video_idx[index]
        clip_frames_video_idx = self.frame_video_idx[start_idx:end_idx + 1]
        need_border_padding = clip_length - len(clip_frames_video_idx)

        if need_border_padding > 0:
            if start_idx == 0:
                clip_frames_video_idx = [clip_frames_video_idx[0]] * need_border_padding + clip_frames_video_idx
            else:
                clip_frames_video_idx = clip_frames_video_idx + [clip_frames_video_idx[-1]] * need_border_padding

        all_frames_same_video = np.array(clip_frames_video_idx) - main_frame_video_idx
        offset = np.sum(all_frames_same_video)

        if all_frames_same_video[0] != 0 and all_frames_same_video[-1] != 0:    # Extreme condition that is not likely to happen
            print('The video is too short or the context frame number is too large!')
            raise NotImplementedError

        if need_border_padding == 0 and offset == 0:
            idx = [x for x in range(start_idx, end_idx + 1)]
            return idx
        else:
            if self.mode == 'last':
                if need_border_padding > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                idx = [x for x in range(start_idx - offset, end_idx + 1)]
                idx = [idx[0]] * np.maximum(np.abs(offset), need_border_padding) + idx
                return idx
            else:   # Means that self.mode == "middle"
                if need_border_padding > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                if offset > 0:
                    idx = [x for x in range(start_idx, end_idx - offset + 1)]
                    idx = idx + [idx[-1]] * np.abs(offset)
                    return idx
                elif offset < 0:
                    idx = [x for x in range(start_idx - offset, end_idx + 1)]
                    idx = [idx[0]] * np.abs(offset) + idx
                    return idx
                if need_border_padding > 0:
                    if start_idx == 0:
                        idx = [x for x in range(start_idx, end_idx + 1)]
                        idx = [idx[0]] * need_border_padding + idx
                        return idx
                    else:
                        idx = [x for x in range(start_idx, end_idx + 1)]
                        idx = idx + [idx[-1]] * need_border_padding
                        return idx

    def resize_batch(self, frame_idx_range, flows_batch):
        # img_batch_resized = []
        flows_batch_resized = []
        for i in range(len(frame_idx_range)):
            cur_flow = flows_batch[i]
            cur_flow = cv2.resize(np.transpose(cur_flow, [1, 2, 0]), (self.patch_size, self.patch_size))
            flows_batch_resized.append(np.transpose(cur_flow, [2, 0, 1]))
        flows_batch = np.array(flows_batch_resized)
        return flows_batch

    def __getitem__(self, index):
        return self.flows_batch_all[index], self.flows_batch_index[index]

    def gendata(self):
        self.flows_batch_all = []
        self.flows_batch_index = []
        caif = 5
        for index in tqdm(range(len(self.frame_addresses))):
            # print(index)
            if self.train and self.dataset_name =="shanghaitech":  ### 只用 十分之一的数据训练
                if index % caif != 0:
                    continue
            frame_idx_range = self.get_sequence(index=index)

            flows_batch = []
            object_id = []
            for i in frame_idx_range:
                # [h,w,c] -> [c,h,w] BGR
                cur_flow = np.transpose(np.load(self.frame_addresses_flows[i]), [2, 0, 1])
                flows_batch.append(cur_flow)

            flows_batch = np.array(flows_batch)

            if self.all_bboxes is not None:
                # cropping
                if len(self.all_bboxes[index]) > 0:
                    # img_batch = get_foreground(img=img_batch, bboxes=self.all_bboxes[index], patch_size=self.patch_size)
                    flows_batch = get_foreground(img=flows_batch, bboxes=self.all_bboxes[index], patch_size=self.patch_size)
                    ## 8帧  
                    # img_batch = get_foreground_8(img=img_batch, bboxes=self.all_bboxes[index], patch_size=self.patch_size)
                    # img_batch,object_id = get_foreground_track(img=img_batch, bboxes=self.all_bboxes[index], patch_size=self.patch_size)
                    # flows_batch,object_id = get_foreground_track(img=flows_batch, bboxes=self.all_bboxes[index], patch_size=self.patch_size)
                else:  # 没有检测出物体 直接把原图返回
                    flows_batch = self.resize_batch(frame_idx_range, flows_batch)
                    flows_batch = flows_batch[None]
                    object_id = []
      
            flows_batch = torch.from_numpy(flows_batch)  # [num_bboxes, sequence_length, C, patch_size, patch_size]

            for flow in flows_batch:
                self.flows_batch_all.append(flow)
                self.flows_batch_index.append(index)
        return self.flows_batch_all

transform = transforms.Compose([
    transforms.ToTensor(),
])

def normalize(img):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    if len(img.shape) == 4:
        return img.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    elif len(img.shape) == 5:
        return img.sub_(mean[None, None, :, None, None]).div_(std[None, None, :, None, None])



if __name__ == '__main__':
    # for i in range(10):
    #     indices = np.random.choice(10, 1, replace=False)
    #     print(indices)

    from torch.utils.data import DataLoader
    root = "D:\project_python\Accurate-Interpretable-VAD\data"
    flowdataset_name = "shanghaitech"
    all_bboxes_train = np.load(os.path.join(root, flowdataset_name, '%s_bboxes_train.npy' % flowdataset_name),
                            allow_pickle=True)
    all_bboxes_test = np.load(os.path.join(root, flowdataset_name, '%s_bboxes_test.npy' % flowdataset_name),
                              allow_pickle=True)
    loader = dict()
     # dataset_flow = 
    datasetflow_train = VideoDatasetWithFlows(dataset_name=flowdataset_name, root=root, patch_size=112,
                                          train=True, sequence_length=0, all_bboxes=all_bboxes_train, normalize=True)
    datasetflow_test = VideoDatasetWithFlows(dataset_name=flowdataset_name, root=root, patch_size=112,
                                         train=False, sequence_length=0, all_bboxes=all_bboxes_test, normalize=True,mode = "last")
    loader_args = {'batch_size': 32, 'num_workers': 4, 'pin_memory': True}
    loader["train"] = DataLoader(datasetflow_train, **loader_args, shuffle=(True))
    loader["test"] = DataLoader(datasetflow_test, **loader_args, shuffle=(False))

    for batch in loader["train"]:
        print(batch.shape)
        