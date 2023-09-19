import pandas as pd 
import numpy as np 

from transformers import AutoTokenizer
import torch

data_path = '/usr/src/kaggle_/kaggle_llm_code/kaggle_dataset/kaggle-llm-science-exam'
df_train = pd.read_csv(data_path+'/'+'train.csv')
df_train = df_train.drop(columns="id")
df_train.shape

extra_train_df = pd.read_csv('/usr/src/kaggle_/kaggle_llm_code/kaggle_dataset/kaggle-llm-science-exam/extra_train_set.csv')

df_train = pd.concat([df_train,extra_train_df]).reset_index(drop=True)

option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
index_to_option = {v: k for k,v in option_to_index.items()}

tokenizer = AutoTokenizer.from_pretrained(deberta_v3_large)

def preprocess(example):
    first_sentence = [example['prompt']] * 5
    second_sentences = [example[option] for option in 'ABCDE']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation=True)
    tokenized_example['label'] = option_to_index[example['answer']]
    
    return tokenized_example

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch
    
tokenizer = AutoTokenizer.from_pretrained(deberta_v3_large)

dataset = Dataset.from_pandas(df_train)