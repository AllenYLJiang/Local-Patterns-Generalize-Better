import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch.nn.functional as F
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
# import sys
# sys.path.append('/media/yons/C50F62CB0B8B9B1C')
from demo_batch import batch_process
import os
import cv2
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(name="blip2", model_type="pretrain", is_eval=True,
                                                     device=device)  # , torch_dtype=torch.float16)
nltk.data.path.append('nlp_libs')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

def statement_to_question(sentence):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    # 找到第一个助动词
    first_verb_index = next((i for i, word in enumerate(pos_tags) if word[1].startswith('VB')), None)

    if first_verb_index is not None:
        question = tokens[first_verb_index] + ' ' + ' '.join(tokens[:first_verb_index] + tokens[first_verb_index + 1:]) + '?'
    else:
        question = sentence + '?'

    question = question[0].upper() + question[1:]
    return question

def split_sentence_nltk(sentence):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    # Initialize split points
    split_point = None

    # Look for conjunctions (CC) or punctuation (,) as potential split points
    for i, (word, pos) in enumerate(pos_tags):
        if pos == 'CC' or word == ',':
            split_point = i
            break

    # If a split point is found, split the sentence
    if split_point:
        left_part = TreebankWordDetokenizer().detokenize(tokens[:split_point])
        right_part = TreebankWordDetokenizer().detokenize(tokens[split_point + 1:])
        return left_part.strip(), right_part.strip()

    return sentence, ""  # Return the full sentence if no split is found

def sentence_weighted_visual_features(curr_batch_bbox_list, curr_batch_img_name_list, template_caption_of_curr_traj):
    with torch.no_grad():
        curr_batch_img_cat_img_name_list, curr_batch_img_cat_list = [], []
        # dump the crops
        if not os.path.exists('dump'):
            os.mkdir('dump')
        for ele_idx in range(len(curr_batch_bbox_list)):
            left, top, width, height = curr_batch_bbox_list[ele_idx][0], curr_batch_bbox_list[ele_idx][1], \
                                       curr_batch_bbox_list[ele_idx][2], curr_batch_bbox_list[ele_idx][3]
            img_path = curr_batch_img_name_list[ele_idx]
            curr_img = cv2.imread(img_path)
            # expand to cater for neighoring bicycles
            left = int(max([0, left - width * 0.33]))
            right = int(min([curr_img.shape[1] - 1, left + width * 1.67]))
            top = int(max([0, top - height * 0.33]))
            bottom = int(min([curr_img.shape[0] - 1, top + height * 1.67]))
            width = right - left
            height = bottom - top

            curr_crop = curr_img[top: top + height, left: left + width, :]
            cv2.imwrite(os.path.join('dump', img_path), curr_crop)
            curr_batch_img_cat_img_name_list.append(os.path.join('dump', img_path))
            # load sample image
            raw_image = Image.open(img_path).convert("RGB")
            raw_image = raw_image.crop((left, top, left+width, top+height))
            # prepare the image
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device) #, torch.float16)
            curr_batch_img_cat_list.append(image)

        # generate sentence
        answer = model.generate({"image": torch.cat(curr_batch_img_cat_list, dim=0), "prompt": " "})[0]
        for answer_idx in range(1, len(answer)):
            if answer[answer_idx] == '':
                answer[answer_idx] = answer[answer_idx - 1]
        # model.generate({"image": image, "prompt": "Question: What action? Answer: "})

        # correct answer based on captions from high-resolution images
        if len(template_caption_of_curr_traj) > 0 and ('walking' not in template_caption_of_curr_traj['caption'] and 'walk' not in template_caption_of_curr_traj['caption'] and 'walks' not in template_caption_of_curr_traj['caption'] and 'standing' not in template_caption_of_curr_traj['caption'] and 'stand' not in template_caption_of_curr_traj['caption'] and 'stands' not in template_caption_of_curr_traj['caption'] and 'sitting' not in template_caption_of_curr_traj['caption'] and 'sit' not in template_caption_of_curr_traj['caption'] and 'sits' not in template_caption_of_curr_traj['caption']):
            print(template_caption_of_curr_traj['caption'])
            input_str = template_caption_of_curr_traj['caption'].split('.')[0]
            input_str = input_str[0].lower() + input_str[1:]
            short_answer = batch_process(curr_batch_img_cat_img_name_list, "<img>{}</img>" + statement_to_question( input_str ).replace('?', ':'))
            if '' in short_answer:
                short_answer = batch_process(curr_batch_img_cat_img_name_list, "<img>{}</img>" + statement_to_question( split_sentence_nltk(input_str)[0] ).replace('?', ':'))

            # In each sentence, short_answer denotes whether curr person is doing the behavior as previous high-resolution counterpart
            for answer_idx in range(len(answer)):
                if (short_answer[answer_idx][:3] == 'Yes' or short_answer[answer_idx][:3] == 'yes') and \
                    (curr_batch_bbox_list[answer_idx][2] <= template_caption_of_curr_traj['width'] and curr_batch_bbox_list[answer_idx][3] <= template_caption_of_curr_traj['height']):
                    answer[answer_idx] = template_caption_of_curr_traj['caption']
            # if not sure, do not update template_caption_of_curr_traj, if inserted "walking" it will not update
            for short_answer_idx in range(len(short_answer)):
                if short_answer[short_answer_idx] == '':
                    answer[short_answer_idx] += 'This is a special symbol: walking'

        cross_attention_output_features = model.forward_image(torch.cat(curr_batch_img_cat_list, dim=0))[0]

        curr_sentence_tokens = model.tokenizer([x.split('This is a special symbol: walking')[0] for x in answer], padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(device)

        curr_sentence_features = model.forward_text(curr_sentence_tokens)

        weighted_sum_of_visual_features_list = []
        cross_modality_similarities_array = np.zeros((cross_attention_output_features.shape[0]))
        for ele_idx in range(cross_attention_output_features.shape[0]):
            weighted_sum_of_visual_features = torch.zeros((cross_attention_output_features.shape[2])).to(device)
            for visual_idx in range(cross_attention_output_features.shape[1]):
                cos_sim = F.cosine_similarity(cross_attention_output_features[ele_idx][visual_idx], curr_sentence_features[ele_idx], dim=0) # .item()
                weighted_sum_of_visual_features += cos_sim * cross_attention_output_features[ele_idx][visual_idx]
                cross_modality_similarities_array[ele_idx] += cos_sim
            weighted_sum_of_visual_features_list.append(weighted_sum_of_visual_features.cpu().numpy())
            if len(template_caption_of_curr_traj) > 0:
                if (answer[ele_idx] == template_caption_of_curr_traj['caption']) or (template_caption_of_curr_traj['caption'] in answer[ele_idx]):
                    weighted_sum_of_visual_features_list[-1] = template_caption_of_curr_traj['feature_memorized']

        del cross_attention_output_features, curr_sentence_features
        torch.cuda.empty_cache()

    return weighted_sum_of_visual_features_list, answer, cross_modality_similarities_array

# example_feature = sentence_weighted_visual_features()



