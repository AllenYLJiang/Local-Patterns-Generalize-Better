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
nltk.data.path.append('/path/to/nlplibs')
import json
import copy
import random
import joblib

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
blip2_model, vis_processors, _ = load_model_and_preprocess(name="blip2", model_type="pretrain", is_eval=True,
                                                     device=device)  # , torch_dtype=torch.float16)

MODEL_ID = "/path/to/Qwen-VL"

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

def batch_process_different_questions(images, input_str = ["<img>{}</img>What is the subject doing and where is the subject:"]):
    queries = [
        input_str[i].format(images[i]) for i in range(len(images)) # "<img>{}</img>What is the subject doing and where is the subject:".format(i) for i in images # "<img>{}</img>What is the subject doing and where is the subject:".format(i) for i in images
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


def statement_to_question(sentence):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

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

def have_no_common_words(sentence1, sentence2):
    words1 = set(sentence1.lower().split())
    words2 = set(sentence2.lower().split())
    return words1.isdisjoint(words2)

####################################################### Object detection #######################################################
train_img_dir = '/path/to/training_imgs'
val_img_dir = '/path/to/validation_imgs'

train_bbox_file = '/path/to/training_bboxes.json'
val_bbox_file = '/path/to/validation_bboxes.json'

dst_dir = '/path/to/train_data_SMM'

train_bbox_info = json.load(open(train_bbox_file))
val_bbox_info = json.load(open(val_bbox_file))

img_list = train_bbox_info['images'] + val_bbox_info['images']
bbox_list = train_bbox_info['annotations'] + val_bbox_info['annotations']
human_bbox_list = [x for x in bbox_list if x['category_id'] == 1]

batch_size = 24

captions_list = ['Is the person standing:', 'Is the person sitting:', 'Is the person pushing a stroller:', 'Is the person riding motorcycle',
                 'Is the person fighting:', 'Is the person walking:', 'Is the person running:', 'Is the person riding a bicycle:',
                 'Is the person falling down:', 'Is the person lying on the ground:', 'Is the person squatting:']

curr_batch_img_cat_list, images = [], []

dataset_training_SMM = []

for curr_bbox_info_idx in range(len(human_bbox_list)):
    curr_bbox_info = human_bbox_list[curr_bbox_info_idx]
    assert(len([x for x in img_list if ('%012d'%curr_bbox_info['image_id']) in x['file_name']]) == 1)
    curr_bbox_img_name = [x for x in img_list if ('%012d'%curr_bbox_info['image_id']) in x['file_name']][0]
    if os.path.exists(os.path.join(train_img_dir, curr_bbox_img_name['file_name'])):
        curr_bbox_img = cv2.imread(os.path.join(train_img_dir, curr_bbox_img_name['file_name']))
    else:
        curr_bbox_img = cv2.imread(os.path.join(val_img_dir, curr_bbox_img_name['file_name']))

    curr_bbox_coord = curr_bbox_info['bbox']
    curr_bbox_left, curr_bbox_top, curr_bbox_width, curr_bbox_height = curr_bbox_coord
    curr_bbox_left = max([curr_bbox_left - curr_bbox_width/2, 0])
    curr_bbox_top = max([curr_bbox_top - curr_bbox_height/2, 0])
    curr_bbox_right = min([curr_bbox_left + curr_bbox_width*2, curr_bbox_img.shape[1]-1])
    curr_bbox_bottom = min([curr_bbox_top + curr_bbox_height*2, curr_bbox_img.shape[0]-1])

    curr_bbox_region = curr_bbox_img[int(curr_bbox_top):int(curr_bbox_bottom), int(curr_bbox_left):int(curr_bbox_right), :]

    cv2.imwrite('dump/dump_' + str(curr_bbox_info_idx) + '.jpg', curr_bbox_region)
    images.append('dump/dump_' + str(curr_bbox_info_idx) + '.jpg')# load sample image
    raw_image = Image.open('dump/dump_' + str(curr_bbox_info_idx) + '.jpg').convert("RGB")
    # prepare the image
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device) #, torch.float16)
    curr_batch_img_cat_list.append(image)

    if len(curr_batch_img_cat_list) >= batch_size:
        with torch.no_grad():
            answer = batch_process(images, "<img>{}</img>What is the person doing:")
            answer = [(x[0].lower() + x[1:]) for x in answer]
            answer = [(x.split('.')[0] if x[-1]=='.' else x) for x in answer]

            # the questions with positive answers:
            questions = []
            for answer_ele in answer:
                if 'is' not in answer_ele:
                    questions.append('Is the person ' + answer_ele + ':')
                else:
                    questions.append(statement_to_question( answer_ele ).replace('?', ':'))
            positive_answers = ['yes'] * len(questions)

            # # the questions with negative answers:
            # short_answer = batch_process(curr_batch_img_cat_img_name_list, "<img>{}</img>" + statement_to_question( input_str ).replace('?', ':'))
            # if '' in short_answer:
            #     short_answer = batch_process(curr_batch_img_cat_img_name_list, "<img>{}</img>" + statement_to_question( split_sentence_nltk(input_str)[0] ).replace('?', ':'))
            # For each answer, random select one sentence from captions_list without common word and create a question
            negative_questions = []
            for answer_ele in answer:
                if len([x for x in captions_list if have_no_common_words(x, answer_ele.replace('is', '').replace('person', ''))]) > 0:
                    negative_questions.append(random.choice([x for x in captions_list if have_no_common_words(x, answer_ele.replace('is', '').replace('person', '').replace('the', ''))]))
                else:
                    negative_questions.append(random.choice(captions_list))
            negative_answers = batch_process_different_questions(images, [('<img>{}</img>'+x) for x in negative_questions])
            negative_answers = [('no' if (x[:2] == 'no' or x[:2] == 'No') else '') for x in negative_answers]

            # visual tokens
            cross_attention_output_features = blip2_model.forward_image(torch.cat(curr_batch_img_cat_list, dim=0))[0]

            curr_sentence_tokens = blip2_model.tokenizer([x for x in questions], padding="max_length",
                                                         truncation=True, max_length=32, return_tensors="pt").to(device)
            curr_sentence_features = blip2_model.forward_text(curr_sentence_tokens)
            vision_tokens, positive_questions_tokens = cross_attention_output_features.cpu().numpy(), curr_sentence_features.cpu().numpy()
            curr_sentence_tokens = blip2_model.tokenizer([x for x in negative_questions], padding="max_length", truncation=True,
                                                         max_length=32, return_tensors="pt").to(device)
            curr_sentence_features = blip2_model.forward_text(curr_sentence_tokens)
            negative_questions_tokens = curr_sentence_features.cpu().numpy()

            del cross_attention_output_features, curr_sentence_features
            torch.cuda.empty_cache()

            curr_batch_img_cat_list, images = [], []

            ############## save positive sequences ##################################################
            positive_sequence_to_SMM = [([positive_questions_tokens[x]] + [y for y in vision_tokens[x]] + [np.ones(768).astype('float32')]) for x in range(len(vision_tokens))]
            negative_sequence_to_SMM = []
            for x in range(len(vision_tokens)):
                if negative_answers[x] == 'no':
                    negative_sequence_to_SMM.append([negative_questions_tokens[x]] + [y for y in vision_tokens[x]] + [np.zeros(768).astype('float32')])

            dataset_training_SMM += positive_sequence_to_SMM
            dataset_training_SMM += negative_sequence_to_SMM

            joblib.dump(dataset_training_SMM, os.path.join(dst_dir, 'SMM_training_batch_ending_at_' + str(curr_bbox_info_idx) + '.pkl'))

            dataset_training_SMM = []

            for dump_img_name in os.listdir('dump/'):
                os.remove(os.path.join('dump/', dump_img_name))

    # if video_name[-4:] != '.avi':
    #     out_file = open(os.path.join(dst_dir, video_name + '.json'), "w")
    #     json.dump(curr_video_features, out_file)
    #     out_file.close()
    # else:
    #     out_file = open(os.path.join(dst_dir, video_name.replace('.avi', '.json')), "w")
    #     json.dump(curr_video_features, out_file)
    #     out_file.close()


