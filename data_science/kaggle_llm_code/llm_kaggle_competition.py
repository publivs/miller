
import sys
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier


# ---------------------------------- 读取数据 ------------------------------------ # 
is_offline = True
if is_offline:
    dir_path = '''data_science/kaggle_llm_code'''
else:
    dir_path = '''/kaggle/input/'''

test = pd.read_csv(f'{dir_path}/llm-detect-ai-generated-text/test_essays.csv')
sub = pd.read_csv(f'{dir_path}/llm-detect-ai-generated-text/sample_submission.csv')
org_train = pd.read_csv(f'{dir_path}/llm-detect-ai-generated-text/train_essays.csv')
train = pd.read_csv(f"{dir_path}/daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=',')

train = train.drop_duplicates(subset=['text'])

train.reset_index(drop=True, inplace=True)
print(test.text.values)
# ------------------------------------ END -------------------------------------- # 


# -------------------------------- tokenizer_cfg -------------------------------------- # 
LOWERCASE = False
VOCAB_SIZE = 30522

# Creating Byte-Pair Encoding tokenizer
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# Adding normalization and pre_tokenizer
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# Adding special tokens and creating trainer instance
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

# Creating huggingface dataset object
dataset = Dataset.from_pandas(test[['text']])
# ------------------------------------- END ------------------------------------------- # 


# ------------------------------ tokenize_functions ------------------------------ # 

def train_corp_iter():
    """
    A generator function for iterating over a dataset in chunks.
    """    
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

# Training from iterator REMEMBER it's training on test set...
raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)

tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=raw_tokenizer,
                unk_token="[UNK]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                mask_token="[MASK]",
            )

tokenized_texts_test = []

# Tokenize test set with new tokenizer
for text in tqdm(test['text'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))


# Tokenize train set
tokenized_texts_train = []

for text in tqdm(train['text'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))


def dummy(text):
    """
    A dummy function to use as tokenizer for TfidfVectorizer. It returns the text as it is since we already tokenized it.
    """
    return text

# Fitting TfidfVectoizer on test set

vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None, strip_accents='unicode'
                            )

vectorizer.fit(tokenized_texts_test)

# Getting vocab
vocab = vectorizer.vocabulary_

print(vocab)


# Here we fit our vectorizer on train set but this time we use vocabulary from test fit.
vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                            analyzer = 'word',
                            tokenizer = dummy,
                            preprocessor = dummy,
                            token_pattern = None, strip_accents='unicode')

tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)

del vectorizer
gc.collect()

# -------------------------------------- generate_models ------------------------------------------- #

y_train = train['label'].values

print(tf_train.shape)

bayes_model = MultinomialNB(alpha=0.02)
sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")

ensemble = VotingClassifier(estimators=[('sgd', sgd_model), ('nb', bayes_model)],
                            weights=[0.7, 0.3], voting='soft', n_jobs=-1)
ensemble.fit(tf_train, y_train)

gc.collect()
# --------------------------------------------- END ------------------------------------------------- #


final_preds = ensemble.predict_proba(tf_test)[:,1]

sub['generated'] = final_preds
sub.to_csv('submission.csv', index=False)
