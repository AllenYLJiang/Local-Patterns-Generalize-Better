from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
import os
import numpy as np
from transformers.generation import GenerationConfig
torch.manual_seed(1234)
import cv2
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch.nn.functional as F
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.data.path.append('/media/yons/PortableSSD/TIP2024_code')
import json

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
blip2_model, vis_processors, _ = load_model_and_preprocess(name="blip2", model_type="pretrain", is_eval=True,
                                                     device=device)  # , torch_dtype=torch.float16)

MODEL_ID = "/media/yons/C50F62CB0B8B9B1C/TIP2024_code/facebook/Qwen-VL"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)
tokenizer.padding_side = 'left'
tokenizer.pad_token_id = tokenizer.eod_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cuda",
    trust_remote_code=True,
    bf16=True
).eval()

####################################################### functions ######################################################################
def sliding_window_crop(image, window_size=(224, 224), step_size=(112, 112)):
    """
    Perform sliding window crop on an image with the specified window size and step size.

    :param image: Input image as a NumPy array (e.g., loaded using cv2.imread)
    :param window_size: Tuple (height, width) specifying the size of the crop window
    :param step_size: Tuple (height, width) specifying the step size of the sliding window
    :return: List of cropped images
    """
    crops = []
    img_height, img_width = image.shape[:2]
    window_height, window_width = window_size
    step_height, step_width = step_size

    for y in range(0, img_height - window_height + step_height, step_height):
        for x in range(0, img_width - window_width + step_width, step_width):
            # Adjust the window if it goes beyond the image boundaries
            crop = image[y:min(y + window_height, img_height), x:min(x + window_width, img_width)]
            crops.append(crop)

    return crops

def batch_process(images, input_str = "<img>{}</img>What is the subject doing and where is the subject:"):
    queries = [
        input_str.format(i) for i in images # "<img>{}</img>What is the subject doing and where is the subject:".format(i) for i in images # "<img>{}</img>What is the subject doing and where is the subject:".format(i) for i in images
    ]

    input_tokens = tokenizer(queries, return_tensors='pt', padding='longest')
    input_ids = input_tokens.input_ids
    input_len = input_ids.shape[-1]
    attention_mask = input_tokens.attention_mask

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask = attention_mask.cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=30,
            length_penalty=0,
            num_return_sequences=1,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
        )
    answers = [
        tokenizer.decode(o[input_len:].cpu(), skip_special_tokens=True).strip() for o in outputs
    ]
    end = time.time()
    return answers
    # print("took: ", end - start)
    # for a in answers:
    #     print(a)

# images = [
#     '/media/yons/PortableSSD/TIP2024_code/test_008.jpg',
#     '/media/yons/PortableSSD/TIP2024_code/Qwen_VL_master/dump/dump_12.jpg',
#     '/media/yons/PortableSSD/TIP2024_code/Qwen_VL_master/dump/dump_0.jpg',
#     '/media/yons/PortableSSD/TIP2024_code/Qwen_VL_master/dump/dump_5.jpg',
#     '/media/yons/PortableSSD/TIP2024_code/Qwen_VL_master/dump/dump_6.jpg',
# ]
# batch_process(images, "<img>{}</img>Please describe the contexts in details:")

####################################################### Object detection with Qwen-VL #######################################################
trainval_dir = '/media/yons/C50F62CB0B8B9B1C/STG-NF/data/ShanghaiTech/images/frames_part'
dst_dir = '/media/yons/PortableSSD/TIP2024_code/text_weighted_visual_features/End_to_end_VLM_feature_extraction'

for video_name in sorted(os.listdir(trainval_dir), key=lambda x:x.split('.')[0])[1:]:
    print(video_name)
    if video_name[-4:] != '.avi':
        curr_video_frame_list = sorted(os.listdir(os.path.join(trainval_dir, video_name)), key=lambda x:int(x.split('.')[0]))
        curr_video_num_frames = len(curr_video_frame_list)
    else:
        cap = cv2.VideoCapture(os.path.join(trainval_dir, video_name))
        video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        curr_video_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    curr_video_features = []
    for curr_frame_idx in range(curr_video_num_frames):
        print('frame index: ' + str(curr_frame_idx))
        if video_name[-4:] != '.avi':
            curr_frame = cv2.imread(os.path.join(trainval_dir, video_name, curr_video_frame_list[curr_frame_idx]))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)
            success_or_no, curr_frame = cap.read()
            if success_or_no is False:
                continue
        # crop sliding window
        video_width, video_height, crop_size_height, crop_size_width = curr_frame.shape[1], curr_frame.shape[0], 320, 320
        curr_frame_sliding_window_cropped_regions = sliding_window_crop(curr_frame, window_size=(crop_size_height, crop_size_width), step_size=(int(crop_size_height/2), int(crop_size_width/2)))
        ################################## detect objects from each sliding window ########################################################
        images, curr_batch_img_cat_list = [], []
        for single_crop_idx in range(len(curr_frame_sliding_window_cropped_regions)):
            cv2.imwrite('dump/dump_' + str(single_crop_idx) + '.jpg', curr_frame_sliding_window_cropped_regions[single_crop_idx])
            images.append('dump/dump_' + str(single_crop_idx) + '.jpg')# load sample image
            raw_image = Image.open('dump/dump_' + str(single_crop_idx) + '.jpg').convert("RGB")
            # prepare the image
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device) #, torch.float16)
            curr_batch_img_cat_list.append(image)

        with torch.no_grad():
            answer = batch_process(images, "<img>{}</img>Please describe the contexts in details:")

            cross_attention_output_features = blip2_model.forward_image(torch.cat(curr_batch_img_cat_list, dim=0))[0]

            curr_sentence_tokens = blip2_model.tokenizer([x for x in answer], padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(device)

            curr_sentence_features = blip2_model.forward_text(curr_sentence_tokens)

            weighted_sum_of_visual_features_list = []
            cross_modality_similarities_array = np.zeros((cross_attention_output_features.shape[0]))
            for ele_idx in range(cross_attention_output_features.shape[0]):
                weighted_sum_of_visual_features = torch.zeros((cross_attention_output_features.shape[2])).to(device)
                for visual_idx in range(cross_attention_output_features.shape[1]):
                    cos_sim = F.cosine_similarity(cross_attention_output_features[ele_idx][visual_idx], curr_sentence_features[ele_idx], dim=0) # .item()
                    weighted_sum_of_visual_features += cos_sim * cross_attention_output_features[ele_idx][visual_idx]
                    cross_modality_similarities_array[ele_idx] += cos_sim
                weighted_sum_of_visual_features_list.append(weighted_sum_of_visual_features.cpu().numpy())

            cross_attention_output_features_backup, curr_sentence_features_backup = cross_attention_output_features.cpu().numpy(), curr_sentence_features.cpu().numpy()
            del cross_attention_output_features, curr_sentence_features
            torch.cuda.empty_cache()

        for single_crop_idx in range(len(curr_frame_sliding_window_cropped_regions)):
            curr_object_dict = {
                'image_id': '%04d' % int(curr_frame_idx) + '.jpg',
                'category_id': 1,
                'keypoints': weighted_sum_of_visual_features_list[single_crop_idx].tolist(),
                'score': 1.0,
                'box of object in crop': [],
                'sentence': answer[single_crop_idx],
                'the crop the box belongs to': '',  # curr_frame_sliding_window_cropped_regions[single_crop_idx],
                'visual features': [], # cross_attention_output_features_backup[single_crop_idx].tolist(),
                'sentence features': [], # curr_sentence_features_backup[single_crop_idx].tolist(),
                'the index of crop that the box belongs to': single_crop_idx
            }
            curr_video_features.append(curr_object_dict)

    if video_name[-4:] != '.avi':
        out_file = open(os.path.join(dst_dir, video_name + '.json'), "w")
        json.dump(curr_video_features, out_file)
        out_file.close()
    else:
        out_file = open(os.path.join(dst_dir, video_name.replace('.avi', '.json')), "w")
        json.dump(curr_video_features, out_file)
        out_file.close()


