import pandas as pd 
import sys
import gc
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from transformers import LlamaTokenizer, AutoModelForCausalLM

data_path = 'D:\kaggle_\kaggle_llm_code\kaggle-llm-science-exam'

train_df = pd.read_csv(data_path + '/'+'train.csv',index_col= 'id')

test_df = pd.read_csv(data_path + '/' +'test.csv',index_col='id')

model_name = '/kaggle/input/llama2-7b-stem'

tokenizer = LlamaTokenizer.from_pretrained(model_name)
    
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )