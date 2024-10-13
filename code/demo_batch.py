from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
import os

MODEL_ID = "Qwen-VL"

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

def batch_process(images, input_str = ""):
    queries = [
        input_str.format(i) for i in images 
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