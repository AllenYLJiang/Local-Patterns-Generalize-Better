import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch.nn.functional as F
from inference_blip2_batched_test_text_conditioned_feature_extraction import sentence_weighted_visual_features
import os
import json
import cv2
import numpy as np

# test_video_names_list = [x.split('\n')[0].split('/')[-1] for x in open('/home/yons/Downloads/yolov7-main/run_on_test_frames.sh').readlines()]
# test_jsons_list = [('exp' + str(x)) for x in range(8, 115)]

train_imgs_dir = '/path/to/ShanghaiTech/images/frames_part'
dst_train_json_dir = 'test_LLM_weighted_features_text_conditioned_feature_extraction'
batch_size = 24
det_thresh = 0.3

for train_video_name in sorted([x for x in os.listdir(train_imgs_dir) if x[-4:] != '.avi'], key=lambda x:int(x.split('.')[0])): # [85:86]:
    print(train_video_name)
    curr_video_tracked_result = open('datasets/Shanghaitech/' + train_video_name + '.txt').readlines()
    video_width = cv2.imread(os.path.join(train_imgs_dir, train_video_name, os.listdir(os.path.join(train_imgs_dir, train_video_name))[0])).shape[1] # cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cv2.imread(os.path.join(train_imgs_dir, train_video_name, os.listdir(os.path.join(train_imgs_dir, train_video_name))[0])).shape[0] # cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # organize tracked results into a dict
    curr_video_tracked_objs_dict = {}
    for curr_video_tracked_result_line in curr_video_tracked_result:
        curr_object_frame_idx = int(curr_video_tracked_result_line.split(',')[0])
        curr_object_id = int(curr_video_tracked_result_line.split(',')[1])
        curr_object_left = int(round(float(curr_video_tracked_result_line.split(',')[2])))
        curr_object_top = int(round(float(curr_video_tracked_result_line.split(',')[3])))
        curr_object_width = int(round(float(curr_video_tracked_result_line.split(',')[4])))
        curr_object_height = int(round(float(curr_video_tracked_result_line.split(',')[5])))
        curr_object_left = min([max([curr_object_left, 0]), video_width - 1])
        curr_object_top = min([max([curr_object_top, 0]), video_height - 1])
        curr_object_width = min([max([curr_object_width, 0]), video_width - 1 - curr_object_left])
        curr_object_height = min([max([curr_object_height, 0]), video_height - 1 - curr_object_top])
        curr_object_conf = float(curr_video_tracked_result_line.split(',')[6])
        if curr_object_id not in curr_video_tracked_objs_dict:
            curr_video_tracked_objs_dict[curr_object_id] = {
                int(curr_object_frame_idx - 1): {"left": curr_object_left, "top": curr_object_top, "width": curr_object_width, "height": curr_object_height}
            }
        else:
            curr_video_tracked_objs_dict[curr_object_id][int(curr_object_frame_idx - 1)] = \
                {"left": curr_object_left, "top": curr_object_top, "width": curr_object_width, "height": curr_object_height}

    curr_video_updated_json = []
    for traj_id in curr_video_tracked_objs_dict:
        curr_traj_all_frames_boxes = curr_video_tracked_objs_dict[traj_id]

        curr_video_json = []
        frame_list = [('%03d' % int(x) + '.jpg') for x in curr_traj_all_frames_boxes]  # if x[:len(train_video_name.split('.')[0] + '_')] == (train_video_name.split('.')[0] + '_')]
        frame_list = sorted(frame_list, key=lambda x: int(x.split('.')[0]))
        for frame_name in frame_list:
            curr_traj_curr_frame_boxes = curr_traj_all_frames_boxes[int(frame_name.split('.')[0])]

            left, right, top, bottom = curr_traj_curr_frame_boxes['left'], curr_traj_curr_frame_boxes['left'] + curr_traj_curr_frame_boxes['width'], \
                curr_traj_curr_frame_boxes['top'], curr_traj_curr_frame_boxes['top'] + curr_traj_curr_frame_boxes['height']

            curr_video_json.append({
                'image_id': frame_name,
                'category_id': 1,
                'keypoints': [],
                'score': 1.0,
                'box': [left, top, right - left, bottom - top]
            })

        template_caption_of_curr_traj = {'width': 0.0, 'height': 0.0, 'caption': [], 'feature_memorized': []}
        for curr_video_json_ele_idx in range(0, len(curr_video_json), batch_size):
            print('curr_video_json_ele_idx: ' + str(curr_video_json_ele_idx))
            # inside batch
            curr_batch_bbox_list, curr_batch_img_name_list = [], []
            for inner_batch_idx in range(min([batch_size, len(curr_video_json) - curr_video_json_ele_idx])):
                curr_video_json_ele = curr_video_json[curr_video_json_ele_idx + inner_batch_idx]

                curr_frame_idx = int(curr_video_json_ele['image_id'].split('.')[0])
                print('curr_frame_idx: ' + str(curr_frame_idx))
                curr_bbox_left = int(round(curr_video_json_ele['box'][0]))
                curr_bbox_top = int(round(curr_video_json_ele['box'][1]))
                curr_bbox_width = int(round(curr_video_json_ele['box'][2]))
                curr_bbox_height = int(round(curr_video_json_ele['box'][3]))

                curr_frame = cv2.imread(os.path.join(train_imgs_dir, train_video_name, curr_video_json_ele['image_id']))
                cv2.imwrite('dump_' + str(inner_batch_idx) + '.png', curr_frame)
                curr_batch_bbox_list.append([curr_bbox_left, curr_bbox_top, curr_bbox_width, curr_bbox_height])
                curr_batch_img_name_list.append('dump_' + str(inner_batch_idx) + '.png')

            # inputs belong to the same id
            if curr_video_json_ele_idx == 0 or len(template_caption_of_curr_traj['caption']) <= 5:
                example_feature, example_sentences, cross_modality_similarities_array = sentence_weighted_visual_features(curr_batch_bbox_list,
                                                                                                                          curr_batch_img_name_list, {})
            else:
                example_feature, example_sentences, cross_modality_similarities_array = sentence_weighted_visual_features(curr_batch_bbox_list,
                                                                                                                          curr_batch_img_name_list, template_caption_of_curr_traj)
            # if this is the start of trajectory, and resolution is highest
            # if curr_video_json_ele_idx == 0:
            # if revise "-1" also revise if short_answer[-1] == '': in inference-xxx.py
            if (len(example_sentences[np.argmax(cross_modality_similarities_array)]) > 5) and ('walking' not in example_sentences[np.argmax(cross_modality_similarities_array)] and 'standing' not in example_sentences[np.argmax(cross_modality_similarities_array)] and 'sitting' not in example_sentences[np.argmax(cross_modality_similarities_array)]):
                template_caption_of_curr_traj['width'] = curr_batch_bbox_list[np.argmax(cross_modality_similarities_array)][2]
                template_caption_of_curr_traj['height'] = curr_batch_bbox_list[np.argmax(cross_modality_similarities_array)][3]
                template_caption_of_curr_traj['caption'] = example_sentences[np.argmax(cross_modality_similarities_array)]
                template_caption_of_curr_traj['feature_memorized'] = example_feature[np.argmax(cross_modality_similarities_array)]
            # if 'This is a special symbol: walking' in example_sentences[-1]:
            example_sentences = [x.split('This is a special symbol: walking')[0] for x in example_sentences]

            for inner_batch_idx in range(min([batch_size, len(curr_video_json) - curr_video_json_ele_idx])):
                curr_video_json[curr_video_json_ele_idx + inner_batch_idx]['keypoints'] = \
                    example_feature[inner_batch_idx].tolist()
                curr_video_json[curr_video_json_ele_idx + inner_batch_idx]['sentence'] = example_sentences[inner_batch_idx]
                curr_video_updated_json.append(curr_video_json[curr_video_json_ele_idx + inner_batch_idx])

    out_file = open(os.path.join(dst_train_json_dir, train_video_name.split('.')[0] + '.json'), "w")
    json.dump(curr_video_updated_json, out_file)
    out_file.close()

