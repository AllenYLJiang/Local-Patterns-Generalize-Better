# Prepare basis libs

cd code 

download from https://huggingface.co/google-bert/bert-base-uncased and create folder bert-base-uncased

download from https://huggingface.co/Qwen/Qwen-VL/tree/main and place into a new folder "Qwen-VL"  

Note: Qwen-VL 7B can be used or replaced with tradition detectors, without influencing performance 

download the weight file in Q-former and rename to "captioning.pth" 

# Run the code, generate the features  

python update_json_batched_test_text_conditioned_feature_extraction.py 

