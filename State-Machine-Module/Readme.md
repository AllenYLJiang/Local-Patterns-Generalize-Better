# Train stacked state machines 
train_statespace.py --smooth_or_predwithsmoothed_or_predwithunsmoothed train --epochs 300 

# Inference with stacked state machines 
train_statespace.py --smooth_or_predwithsmoothed_or_predwithunsmoothed predwithunsmoothed 

# Predict with SMM 
Step 1: Run inference_SMM_predict_yes_or_no.py to save the embedding of each input image region as a tensor with shape batch_size * 33(=32 vision tokens + 1 question token) * 768(token embedding size).  

For example, a question "Is the person fighting?" is encoded with a 768-dimensional vector which is concatenated the 32*768 embedding of input image. 
Step 2: Save the embeddings in data\data\dataset_name\train_SMM\test 
Step 3: Run train_statespace.py --smooth_or_predwithsmoothed_or_predwithunsmoothed predwithunsmoothed  

The SMM predicts "yes" or "no" based on the question and input image features. 


