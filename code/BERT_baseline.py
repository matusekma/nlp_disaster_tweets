#!/usr/bin/env python
# coding: utf-8
# In[2]:


from preprocessing.preprocessing import text_preprocessing
import random
import numpy as np
from tqdm import tqdm_notebook as tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from sklearn.metrics import accuracy_score, f1_score

from transformers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Since we want to make sure that the results are reproducible every time this kernel runs, we will `seed everything` and fix the randomness

# In[3]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[4]:

seed_everything()

# In[6]:

train = pd.read_csv('../input/train.csv')
print('Training data shape: ', train.shape)
test = pd.read_csv('../input/test.csv')
print('Testing data shape: ', test.shape)
submit = pd.read_csv('../input/sample_submission.csv')

train['text'] = train['text'].apply(lambda x: text_preprocessing(x))
test['text'] = test['text'].apply(lambda x: text_preprocessing(x))

train.drop(["keyword", "location"], axis=1, inplace=True)
test.drop(["keyword", "location"], axis=1, inplace=True)

print(train.head())


# In[9]:


print(train.target.value_counts())


# In[10]:


test.head()


# ## Tokenization

# For NLP tasks, the data are in human readable form i.e. `text` type, so we need to convert in computer readable form. This is where tokenization comes into play. Tokenization involves two steps: breaking words into `tokens` and converting them into vectors. There is a great [blog post](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) about this and using BERT for the first time. An image from the blog is shown below
# ![](http://jalammar.github.io/images/distilBERT/bert-distilbert-tokenization-2-token-ids.png)

# What are `[CLS]` and `[SEP]`?
#
# They are special tokens and `BERT` uses them to mark the beginning(`[CLS]`) and separation/end of sentence(`[SEP]`). In usage, it would look something like this:
# > `[CLS] a visually stunning rumination on love [SEP]`

# We will be using the same tokenization process to tokenize our `train` and `test` data. Let's write the code for that

# In[11]:


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, id, text, label=None):
        """Constructs a InputExample.
        Args:
            id: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.id = id
        self.text = text
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        _, input_ids, input_mask, segment_ids = choices_features[0]
        self.choices_features = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids
        }
        self.label = label


# In[12]:


def read_examples(df, is_training):
    if not is_training:
        df['target'] = np.zeros(len(df), dtype=np.int64)
    examples = []
    for val in df[['id', 'text', 'target']].values:
        examples.append(InputExample(id=val[0], text=val[1], label=val[2]))
    return examples, df


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# In[13]:


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    features = []
    for example_index, example in enumerate(examples):

        text = tokenizer.tokenize(example.text)
        MAX_TEXT_LEN = max_seq_length - 2
        text = text[:MAX_TEXT_LEN]

        choices_features = []

        tokens = ["[CLS]"] + text + ["[SEP]"]
        segment_ids = [0] * (len(text) + 2)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += ([0] * padding_length)
        input_mask += ([0] * padding_length)
        segment_ids += ([0] * padding_length)
        choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        if example_index < 1 and is_training:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example_index))
            logger.info("id: {}".format(example.id))
            logger.info("tokens: {}".format(
                ' '.join(tokens).replace('\u2581', '_')))
            logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            logger.info("input_mask: {}".format(len(input_mask)))
            logger.info("segment_ids: {}".format(len(segment_ids)))
            logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id=example.id,
                choices_features=choices_features,
                label=label
            )
        )
    return features


def select_field(features, field):
    return [
        feature.choices_features[field] for feature in features
    ]


def metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1


# In[14]:


# hyperparameters
max_seq_length = 512
learning_rate = 1e-5
num_epochs = 10
batch_size = 8
patience = 2
file_name = 'model'
bert_model = 'bert-large-uncased'


# In[15]:


logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler('log_model.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


# `BERT` expects three kinds of input: `input_ids`(of tokens), `segment_ids`(to distinguish different sentences), and `input_mask`(to indicate which elements in the sequence are tokens and which are padding elements). The code below gets all three inputs for `train` set. We will be using `bert-base-uncased` in this kernel but you can experiment with other variants as well.

# In[16]:


tokenizer = BertTokenizer.from_pretrained(
    bert_model, do_lower_case=True)


# In[17]:


train_examples, train_df = read_examples(train, is_training=True)
labels = train_df['target'].astype(int).values
train_features = convert_examples_to_features(
    train_examples, tokenizer, max_seq_length, True)
all_input_ids = np.array(select_field(train_features, 'input_ids'))
all_input_mask = np.array(select_field(train_features, 'input_mask'))
all_segment_ids = np.array(select_field(train_features, 'segment_ids'))
all_label = np.array([f.label for f in train_features])


# Similarly for `test` set.

# In[18]:


test_examples, test_df = read_examples(test, is_training=False)
test_features = convert_examples_to_features(
    test_examples, tokenizer, max_seq_length, True)
test_input_ids = torch.tensor(select_field(
    test_features, 'input_ids'), dtype=torch.long)
test_input_mask = torch.tensor(select_field(
    test_features, 'input_mask'), dtype=torch.long)
test_segment_ids = torch.tensor(select_field(
    test_features, 'segment_ids'), dtype=torch.long)


# We will be using `bert-base-uncased` as our base model and add a linear layer with [multisample dropout](https://arxiv.org/abs/1905.09788). This is based on [8th place solution](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100961#latest-593873) of [Jigsaw Competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview).
#

# In[19]:


class NeuralNet(nn.Module):
    def __init__(self, hidden_size=768, num_class=2):
        super(NeuralNet, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model,
                                              output_hidden_states=True,
                                              output_attentions=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, input_ids, input_mask, segment_ids):
        all_hidden_states, all_attentions = self.bert(input_ids, token_type_ids=segment_ids,
                                                      attention_mask=input_mask)[-2:]
        batch_size = input_ids.shape[0]
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(
            13, batch_size, 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(
            13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.fc(dropout(feature))
            else:
                h += self.fc(dropout(feature))
        h = h / len(self.dropouts)
        return h


# We will use `StratifiedKFold` to split our data into `7 folds`. Multifold splitting is a popular validation strategy in kaggle competitions.

# In[20]:


skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
# off: out-of-fold
oof_train = np.zeros((len(train_df), 2), dtype=np.float32)
oof_test = np.zeros((len(test_df), 2), dtype=np.float32)

# In[21]:


for fold, (train_index, valid_index) in enumerate(skf.split(all_label, all_label)):

    # remove this line if you want to train for all 7 folds
    # if fold == 2:
    #   break  # due to kernel time limit

    logger.info(
        '================     fold {}        ==============='.format(fold))

    train_input_ids = torch.tensor(
        all_input_ids[train_index], dtype=torch.long)
    train_input_mask = torch.tensor(
        all_input_mask[train_index], dtype=torch.long)
    train_segment_ids = torch.tensor(
        all_segment_ids[train_index], dtype=torch.long)
    train_label = torch.tensor(all_label[train_index], dtype=torch.long)

    valid_input_ids = torch.tensor(
        all_input_ids[valid_index], dtype=torch.long)
    valid_input_mask = torch.tensor(
        all_input_mask[valid_index], dtype=torch.long)
    valid_segment_ids = torch.tensor(
        all_segment_ids[valid_index], dtype=torch.long)
    valid_label = torch.tensor(all_label[valid_index], dtype=torch.long)

    train = torch.utils.data.TensorDataset(
        train_input_ids, train_input_mask, train_segment_ids, train_label)
    valid = torch.utils.data.TensorDataset(
        valid_input_ids, valid_input_mask, valid_segment_ids, valid_label)
    test = torch.utils.data.TensorDataset(
        test_input_ids, test_input_mask, test_segment_ids)

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False)

    model = NeuralNet()
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-6)
    model.train()

    best_f1 = 0.
    valid_best = np.zeros((valid_label.size(0), 2))

    early_stop = 0
    for epoch in range(num_epochs):
        train_loss = 0.
        for batch in tqdm(train_loader):
            batch = tuple(t.to(device) for t in batch)
            x_ids, x_mask, x_sids, y_truth = batch
            y_pred = model(x_ids, x_mask, x_sids)
            loss = loss_fn(y_pred, y_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)

        model.eval()
        val_loss = 0.
        valid_preds_fold = np.zeros((valid_label.size(0), 2))
        with torch.no_grad():
            for i, batch in tqdm(enumerate(valid_loader)):
                batch = tuple(t.to(device) for t in batch)
                x_ids, x_mask, x_sids, y_truth = batch
                y_pred = model(x_ids, x_mask, x_sids).detach()
                val_loss += loss_fn(y_pred, y_truth).item() / len(valid_loader)
                valid_preds_fold[i * batch_size:(i + 1) * batch_size] = F.softmax(
                    y_pred, dim=1).cpu().numpy()

        acc, f1 = metric(all_label[valid_index],
                         np.argmax(valid_preds_fold, axis=1))
        if best_f1 < f1:
            early_stop = 0
            best_f1 = f1
            valid_best = valid_preds_fold
            torch.save(model.state_dict(), 'model_fold_{}.bin'.format(fold))
        else:
            early_stop += 1
        logger.info(
            'epoch: %d, train loss: %.8f, valid loss: %.8f, acc: %.8f, f1: %.8f, best_f1: %.8f\n' %
            (epoch, train_loss, val_loss, acc, f1, best_f1))
        torch.cuda.empty_cache()

        if early_stop >= patience:
            break

    test_preds_fold = np.zeros((len(test_df), 2))
    valid_preds_fold = np.zeros((valid_label.size(0), 2))
    model.load_state_dict(torch.load('model_fold_{}.bin'.format(fold)))
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(valid_loader)):
            batch = tuple(t.to(device) for t in batch)
            x_ids, x_mask, x_sids, y_truth = batch
            y_pred = model(x_ids, x_mask, x_sids).detach()
            valid_preds_fold[i * batch_size:(i + 1) * batch_size] = F.softmax(
                y_pred, dim=1).cpu().numpy()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            batch = tuple(t.to(device) for t in batch)
            x_ids, x_mask, x_sids = batch
            y_pred = model(x_ids, x_mask, x_sids).detach()
            test_preds_fold[i * batch_size:(i + 1) *
                            batch_size] = F.softmax(y_pred, dim=1).cpu().numpy()
    valid_best = valid_preds_fold
    oof_train[valid_index] = valid_best
    acc, f1 = metric(all_label[valid_index], np.argmax(valid_best, axis=1))
    logger.info('epoch: best, acc: %.8f, f1: %.8f, best_f1: %.8f\n' %
                (acc, f1, best_f1))

    oof_test += test_preds_fold / 7  # uncomment this for 7 folds
    # oof_test += test_preds_fold / 2  # comment this line when training for 7 folds


# In[22]:


logger.info(f1_score(labels, np.argmax(oof_train, axis=1)))
train_df['pred_target'] = np.argmax(oof_train, axis=1)


# In[23]:


train_df.head()


# In[24]:


test_df['target'] = np.argmax(oof_test, axis=1)
logger.info(test_df['target'].value_counts())


# In[25]:


submit['target'] = np.argmax(oof_test, axis=1)
submit.to_csv('submission_3fold.csv', index=False)


# In[26]:


submit.head()


# I trained for all folds offline and selected the models from best folds to make predictions on test-set. The LB score was `0.83640`

# In[27]:


offline_sub = pd.read_csv('../input/bertsubmission/submission.csv')
offline_sub.head()


# In[28]:


offline_sub.to_csv('offline_submission.csv', index=False)
