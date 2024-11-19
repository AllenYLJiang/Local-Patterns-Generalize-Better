"""
Train\Test helper, based on awesome previous work by https://github.com/amirmk89/gepc
"""

import os
import time
import shutil
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from utils_r.data_utils import normalize_pose
import json
from dataset import keypoints17_to_coco18
import copy

def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


class Trainer:
    def __init__(self, args, model, train_loader, test_loader,input_frame=20,
                 optimizer_f=None, scheduler_f=None):
        #####  
        self.input_frame =input_frame
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        # Loss, Optimizer and Scheduler
        self.lossf = torch.nn.MSELoss()
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        elif self.args.optimizer == 'adamx':
            if self.args.lr:
                return optim.Adamax(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adamax(self.model.parameters())
        return optim.SGD(self.model.parameters(), lr=self.args.lr)

    def adjust_lr(self, epoch):
        return adjust_lr(self.optimizer, epoch, self.args.model_lr, self.args.model_lr_decay, self.scheduler)

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = self.args

        path_join = os.path.join(self.args.ckpt_dir, filename)
        torch.save(state, path_join)
        if is_best:
            shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))

    def load_checkpoint(self, filename):
        filename = filename
        try:
            checkpoint = torch.load(filename)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(filename, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))

    def train(self, log_writer=None, clip=100):
        time_str = time.strftime("%b%d_%H%M_")
        checkpoint_filename = time_str + '_checkpoint.pth.tar'
        start_epoch = 0
        num_epochs = self.args.epochs
        self.model.train()
        self.model = self.model.to(self.args.device)
        key_break = False
        for epoch in range(start_epoch, num_epochs):
            if key_break:
                break
            print("Starting Epoch {} / {}".format(epoch + 1, num_epochs))
            pbar = tqdm(self.train_loader)
            for itern, data_arr in enumerate(pbar):
                try:
                    data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
                    score = data[-2].amin(dim=-1)
                    label = data[-1] # 256  float 64 1 ---> 
                    if self.args.model_confidence:
                        samp = data[0]
                    else:
                        samp = data[0][:, :3] ## sample 256 2 24 18
                    # ## transposed ck0706 ####
                    # samp = samp.permute(0,2,1,3) # sample 256 24 18 2
                    # ######################################
                    seq_len = samp.shape[2]
                    samp = samp.permute(0,2,1,3).reshape(-1, seq_len, 768).float()
                    pred = self.model(samp[:,0:self.input_frame,:]) # sample 256 2 24 18  --》256 144
                    if pred is None:
                        continue
                    if self.args.model_confidence: # s
                        pred = pred * score
                    # losses = compute_loss(pred, reduction="mean")["total_loss"]

                    losses = self.lossf(pred, samp[:,self.input_frame:seq_len,:].reshape(-1, (seq_len-self.input_frame )* 768))
                    losses.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    pbar.set_description("Loss: {}".format(losses.item()))
                    # log_writer.add_scalar('NLL Loss', losses.item(), epoch * len(self.train_loader) + itern)

                except KeyboardInterrupt:
                    print('Keyboard Interrupted. Save results? [yes/no]')
                    choice = input().lower()
                    if choice == "yes":
                        key_break = True
                        break
                    else:
                        exit(1)

            self.save_checkpoint(epoch, filename=str(epoch)+ "_ep_" +checkpoint_filename)
            new_lr = self.adjust_lr(epoch)
            print('Checkpoint Saved. New LR: {0:.3e}'.format(new_lr))

    def test(self):
        self.model.eval()
        self.model.to(self.args.device)
        pbar = tqdm(self.test_loader)
        probs = torch.empty(0).to(self.args.device)
        print("Starting Test Eval")
        self.lossf = torch.nn.MSELoss(reduce=False)
        for itern, data_arr in enumerate(pbar):
            data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
            score = data[-2].amin(dim=-1)
            if self.args.model_confidence:
                samp = data[0]
            else:
                samp = data[0][:, :3]
            seq_len = samp.shape[2]
            samp = samp.permute(0,2,1,3).reshape(-1, seq_len, 768).float()
            with torch.no_grad():
                pred = self.model(samp[:,0:self.input_frame,:]) # sample 256 2 24 18  --》256 144
                if pred is None:
                    continue
                if self.args.model_confidence: # 
                    pred = pred * score
                # losses = compute_loss(pred, reduction="mean")["total_loss"]

            losses = self.lossf(pred, samp[:,self.input_frame:seq_len,:].reshape(-1, (seq_len-self.input_frame )* 768))
                    
            # probs = torch.cat((probs, -1 * nll), dim=0)
            probs = torch.cat((probs, -1 * losses.mean(dim=1)), dim=0)
        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
        return prob_mat_np
        
    def inference(self):
        self.model.eval()
        self.model.to(self.args.device)
        pbar = tqdm(self.test_loader)
        probs = torch.empty(0).to(self.args.device)
        print("Starting Inference Eval")
        self.lossf = torch.nn.MSELoss(reduce=False)
        for itern, data_arr in enumerate(pbar):
            data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
            score = data[-2].amin(dim=-1)
            if self.args.model_confidence:
                samp = data[0]
            else:
                samp = data[0][:, :3]
            seq_len = samp.shape[2]
            samp = samp.permute(0,2,1,3).reshape(-1, seq_len, 768).float()
            with torch.no_grad():
                pred = self.model(samp[:,0:self.input_frame,:]) # sample 256 2 24 18  --》256 144
                if pred is None:
                    continue
                if self.args.model_confidence: # 
                    pred = pred * score
                # losses = compute_loss(pred, reduction="mean")["total_loss"]

            losses = self.lossf(pred, samp[:,self.input_frame:seq_len,:].reshape(-1, (seq_len-self.input_frame )* 768))
                    
            # probs = torch.cat((probs, -1 * nll), dim=0)
            probs = torch.cat((probs, -1 * losses.mean(dim=1)), dim=0)
        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
        return prob_mat_np

    def normalize_data_transformed(self, data_transformed):
        unnormalized_pose_sequence = data_transformed.transpose((1, 2, 0))
        earliest_box_size = [np.max(unnormalized_pose_sequence[0, :, 0]) - np.min(unnormalized_pose_sequence[0, :, 0]), \
                             np.max(unnormalized_pose_sequence[0, :, 1]) - np.min(unnormalized_pose_sequence[0, :, 1])]
        earliest_box_size = np.linalg.norm(earliest_box_size)
        sequential_boxes_offsets = [[np.mean(x[:, 0]), np.mean(x[:, 1])] for x in data_transformed.transpose((1, 2, 0))]
        sequential_boxes_offsets = [(np.array(x) - np.array(sequential_boxes_offsets[0])) for x in sequential_boxes_offsets]
        sequential_boxes_offsets_divided_by_box_size = [(x / earliest_box_size) for x in sequential_boxes_offsets]

        normalize_pose_func_args = {'headless': False, 'scale': 0, 'scale_proportional': 1, 'seg_len': 34,
                                    'dataset': 'ShanghaiTech', 'train_seg_conf_th': 0.0, 'specific_clip': None,
                                    'trans_list': None, 'seg_stride': 1, 'vid_path': 'data/ShanghaiTech/test/frames/'}

        normalize_pose_first_return, normalize_pose_second_return_norm_factor, normalize_pose_third_return_mean, normalize_pose_fourth_return_std = \
            normalize_pose(data_transformed.transpose((1, 2, 0))[None, ...], **normalize_pose_func_args)
        data_transformed = normalize_pose_first_return.squeeze(axis=0).transpose(2, 0, 1)

        data_transformed_earliest_box_size = [np.max(data_transformed[0, 0, :]) - np.min(data_transformed[0, 0, :]), \
                                              np.max(data_transformed[1, 0, :]) - np.min(data_transformed[1, 0, :])]
        data_transformed_earliest_box_size = np.linalg.norm(data_transformed_earliest_box_size)
        add_offsets = [[data_transformed_earliest_box_size * x[0], data_transformed_earliest_box_size * x[1]] for x in
                       sequential_boxes_offsets_divided_by_box_size]
        data_transformed[0, :, :] = np.array(
            [(data_transformed[0, x, :] + np.array(add_offsets)[x, 0]) for x in range(len(data_transformed[0, :, :]))])
        data_transformed[1, :, :] = np.array(
            [(data_transformed[1, x, :] + np.array(add_offsets)[x, 1]) for x in range(len(data_transformed[1, :, :]))])

        # also return the coordinates of the first box, unnormalized offsets, normalized offsets, normalize factor, mean and variance
        # top bottom left right
        return data_transformed, \
               [np.min(unnormalized_pose_sequence[0, :, 1]), np.max(unnormalized_pose_sequence[0, :, 1]), np.min(unnormalized_pose_sequence[0, :, 0]), np.max(unnormalized_pose_sequence[0, :, 0])], \
               sequential_boxes_offsets, add_offsets, normalize_pose_second_return_norm_factor, normalize_pose_third_return_mean, normalize_pose_fourth_return_std

    def smooth_pose_sequences_with_state_machine(self, input_file, output_file, seq_len, input_len, conf_gain_thresh):
        print(input_file)
        self.model.eval()
        self.model.to(self.args.device)
        refined_json, refined_json_unnormalized = {}, {}
        for human_id in json.load(open(input_file, 'r')):
            curr_human_17_keypoints_sequence = json.load(open(input_file, 'r'))[human_id]
            curr_human_instant_keys = list(json.load(open(input_file, 'r'))[human_id].keys())
            curr_human_17_keypoints_sequence = [curr_human_17_keypoints_sequence[x]['keypoints'] for x in curr_human_17_keypoints_sequence]
            curr_human_17_keypoints_sequence = np.array(curr_human_17_keypoints_sequence).reshape(len(curr_human_17_keypoints_sequence), 17, 3)

            if len(curr_human_17_keypoints_sequence) < seq_len:
                data_transformed = keypoints17_to_coco18(curr_human_17_keypoints_sequence).transpose(2, 0, 1)
                data_transformed_unnormalized = copy.deepcopy(data_transformed)
                data_transformed = self.normalize_data_transformed(data_transformed)[0]
                data_transformed = data_transformed.transpose(1, 0, 2)
                data_transformed_unnormalized = data_transformed_unnormalized.transpose(1, 0, 2)

                recover_ori_order = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
                curr_human_keypoints_sequence_array = data_transformed[..., recover_ori_order]
                curr_human_keypoints_sequence_dict = {}
                for curr_human_keypoints_sequence_idx in range(curr_human_keypoints_sequence_array.shape[0]):
                    curr_human_keypoints_sequence_dict[curr_human_instant_keys[curr_human_keypoints_sequence_idx]] = {}
                    curr_human_keypoints_sequence_dict[curr_human_instant_keys[curr_human_keypoints_sequence_idx]]['keypoints'] = curr_human_keypoints_sequence_array[curr_human_keypoints_sequence_idx].transpose(1, 0).reshape(51).tolist()
                    curr_human_keypoints_sequence_dict[curr_human_instant_keys[curr_human_keypoints_sequence_idx]]['scores'] = (np.sum(curr_human_keypoints_sequence_array[curr_human_keypoints_sequence_idx][2, :]) * 0.453).astype('float')
                refined_json[human_id] = curr_human_keypoints_sequence_dict # data_transformed.transpose(1, 0, 2).tolist()

                curr_human_keypoints_sequence_array = data_transformed_unnormalized[..., recover_ori_order]
                curr_human_keypoints_sequence_dict = {}
                for curr_human_keypoints_sequence_idx in range(curr_human_keypoints_sequence_array.shape[0]):
                    curr_human_keypoints_sequence_dict[curr_human_instant_keys[curr_human_keypoints_sequence_idx]] = {}
                    curr_human_keypoints_sequence_dict[curr_human_instant_keys[curr_human_keypoints_sequence_idx]]['keypoints'] = curr_human_keypoints_sequence_array[curr_human_keypoints_sequence_idx].transpose(1, 0).reshape(51).tolist()
                    curr_human_keypoints_sequence_dict[curr_human_instant_keys[curr_human_keypoints_sequence_idx]]['scores'] = (np.sum(curr_human_keypoints_sequence_array[curr_human_keypoints_sequence_idx][2, :]) * 0.453).astype('float')
                refined_json_unnormalized[human_id] = curr_human_keypoints_sequence_dict # data_transformed.transpose(1, 0, 2).tolist()
                continue

            for len_24_segment_idx in range(0, len(curr_human_17_keypoints_sequence) - seq_len + 1):
                # normalize for the purpose of feeding into predictor, shape of data_transformed: 3 x 24 x 18
                data_transformed = keypoints17_to_coco18(curr_human_17_keypoints_sequence[len_24_segment_idx:len_24_segment_idx+seq_len, :, :]).transpose(2, 0, 1)
                # data_transformed, coordinates of the first box (top bottom left right), unnormalized offsets, normalize factor, mean and variance
                data_transformed_unnormalized = copy.deepcopy(data_transformed)
                data_transformed, first_box_coord, unnormalized_offsets, normalized_offsets, normalize_factor, normalize_mean, normalize_std = self.normalize_data_transformed(data_transformed)

                if len_24_segment_idx == 0:
                    curr_human_18_keypoints_sequence_fixed = copy.deepcopy(data_transformed[:, :input_len, :].transpose(1, 0, 2))
                    curr_human_18_keypoints_sequence_fixed_unnormalized = copy.deepcopy(data_transformed_unnormalized[:, :input_len, :].transpose(1, 0, 2))

                # 36=18+18 or 18 intersect 18
                pred = self.model(torch.from_numpy(data_transformed[:2, :, :].transpose(1, 0, 2).reshape(-1, seq_len, 36)[:, :input_len, :]).to(self.args.device))
                pred = pred.reshape(-1, (seq_len - input_len), 36)
                pred = pred.reshape(-1, (seq_len - input_len), 2, 18)

                unnormalized_pred = torch.zeros_like(pred)
                for pred_shape_idx in range(pred.shape[3]):
                    unnormalized_pred[:, :, :, pred_shape_idx] = pred[:, :, :, pred_shape_idx] - torch.from_numpy(np.array(normalized_offsets[-(seq_len - input_len):])).reshape(-1, pred.shape[1], pred.shape[2]).to(0)
                unnormalized_pred *= normalize_std[0]
                for pred_time_idx in range(pred.shape[1]):
                    for pred_joint_idx in range(pred.shape[3]):
                        unnormalized_pred[:, pred_time_idx, :, pred_joint_idx] = (unnormalized_pred[:, pred_time_idx, :, pred_joint_idx] + torch.from_numpy(normalize_mean[0]).to(0)) * torch.from_numpy(normalize_factor[:2]).to(0)

                # even if predict 4 instants, only use one most reliable one
                input_part_mean_conf = np.mean(data_transformed[2, :, :][:input_len, :], axis=0).reshape(-1, 18) # a len-18 vector
                input_part_pred = pred.reshape(-1, (seq_len - input_len), 2, 18)[0, 0, :, :].cpu().detach().numpy() # a 2-by-18 tensor
                input_next_moment_conf = data_transformed[2, input_len, :].reshape(-1, 18) # a len-18 vector
                input_next_moment = data_transformed[:, input_len, :] # a 3-by-18 array

                input_part_pred_unnormalized = unnormalized_pred.reshape(-1, (seq_len - input_len), 2, 18)[0, 0, :, :].cpu().detach().numpy() # a 2-by-18 tensor
                input_next_moment_unnormalized = data_transformed_unnormalized[:, input_len, :] # a 3-by-18 array

                if False: # np.mean(input_part_mean_conf) > np.mean(input_next_moment_conf) + conf_gain_thresh:
                    curr_human_18_keypoints_sequence_fixed = np.concatenate((curr_human_18_keypoints_sequence_fixed, np.concatenate((input_part_pred, input_part_mean_conf), axis=0).reshape(-1, 3, 18)), axis=0)
                    curr_human_18_keypoints_sequence_fixed_unnormalized = np.concatenate((curr_human_18_keypoints_sequence_fixed_unnormalized, np.concatenate((input_part_pred_unnormalized, input_part_mean_conf), axis=0).reshape(-1, 3, 18)), axis=0)
                else:
                    curr_human_18_keypoints_sequence_fixed = np.concatenate((curr_human_18_keypoints_sequence_fixed, input_next_moment.reshape(-1, 3, 18)), axis=0)
                    curr_human_18_keypoints_sequence_fixed_unnormalized = np.concatenate((curr_human_18_keypoints_sequence_fixed_unnormalized, input_next_moment_unnormalized.reshape(-1, 3, 18)), axis=0)
                # do not use curr_human_18_keypoints_sequence_fixed to update inputs because we use original inputs to avoid disaters

            for len_24_segment_idx in list(range(-(seq_len - input_len) + 1, 0)):
                data_transformed = np.expand_dims(keypoints17_to_coco18(curr_human_17_keypoints_sequence[len_24_segment_idx, :, :]), axis=0).transpose(2, 0, 1)
                data_transformed_unnormalized = copy.deepcopy(data_transformed)
                data_transformed = self.normalize_data_transformed(data_transformed)[0]
                curr_human_18_keypoints_sequence_fixed = np.concatenate((curr_human_18_keypoints_sequence_fixed, data_transformed.transpose(1, 0, 2)), axis=0)
                curr_human_18_keypoints_sequence_fixed_unnormalized = np.concatenate((curr_human_18_keypoints_sequence_fixed_unnormalized, data_transformed_unnormalized.transpose(1, 0, 2)), axis=0)

            recover_ori_order = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
            curr_human_keypoints_sequence_array = curr_human_18_keypoints_sequence_fixed[..., recover_ori_order]
            curr_human_keypoints_sequence_dict = {}
            for curr_human_keypoints_sequence_idx in range(curr_human_keypoints_sequence_array.shape[0]):
                curr_human_keypoints_sequence_dict[curr_human_instant_keys[curr_human_keypoints_sequence_idx]] = {}
                curr_human_keypoints_sequence_dict[curr_human_instant_keys[curr_human_keypoints_sequence_idx]]['keypoints'] = curr_human_keypoints_sequence_array[curr_human_keypoints_sequence_idx].transpose(1,0).reshape(51).tolist()
                curr_human_keypoints_sequence_dict[curr_human_instant_keys[curr_human_keypoints_sequence_idx]]['scores'] = (np.sum(curr_human_keypoints_sequence_array[curr_human_keypoints_sequence_idx][2, :]) * 0.453).astype('float')

            refined_json[human_id] = curr_human_keypoints_sequence_dict # curr_human_18_keypoints_sequence_fixed.tolist()

            curr_human_keypoints_sequence_array = curr_human_18_keypoints_sequence_fixed_unnormalized[..., recover_ori_order]
            curr_human_keypoints_sequence_dict = {}
            for curr_human_keypoints_sequence_idx in range(curr_human_keypoints_sequence_array.shape[0]):
                curr_human_keypoints_sequence_dict[curr_human_instant_keys[curr_human_keypoints_sequence_idx]] = {}
                curr_human_keypoints_sequence_dict[curr_human_instant_keys[curr_human_keypoints_sequence_idx]]['keypoints'] = curr_human_keypoints_sequence_array[curr_human_keypoints_sequence_idx].transpose(1, 0).reshape(51).tolist()
                curr_human_keypoints_sequence_dict[curr_human_instant_keys[curr_human_keypoints_sequence_idx]]['scores'] = (np.sum(curr_human_keypoints_sequence_array[curr_human_keypoints_sequence_idx][2, :]) * 0.453).astype('float')

            refined_json_unnormalized[human_id] = curr_human_keypoints_sequence_dict  # curr_human_18_keypoints_sequence_fixed.tolist()

        out_file = open(output_file, "w")
        json.dump(refined_json_unnormalized, out_file)
        out_file.close()

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        return checkpoint_state
