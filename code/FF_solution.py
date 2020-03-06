#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[296]:


from __future__ import print_function, division
import itertools
import os

get_ipython().run_line_magic('matplotlib', 'qt')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras

from keras.preprocessing import text, sequence
from keras import utils

import torch
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ### Create plotting function

# In[297]:


plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


# ## Read data

# In[298]:


train = pd.read_csv('../input/train.csv')
print('Training data shape: ', train.shape)
test = pd.read_csv('../input/test.csv')
print('Testing data shape: ', test.shape)
print(train[train.keyword.isnull()])


# ## Preprocessing text

# In[299]:


# TODO - clean text
# Applying a first round of text cleaning techniques
import re, string
def clean_text(text):
    eyes = "[8:=;]"
    nose = "['`\-]?"
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',"<URL>", text)    
    text = re.sub("/"," / ", text)
    text = re.sub('@(\w+)', '<USER>', text)
    text = re.sub('#{eyes}#{nose}[)d]+|[)d]+#{nose}#{eyes}', "<SMILE>", text)
    text = re.sub('#{eyes}#{nose}p+', "<LOLFACE>", text)
    text = re.sub('#{eyes}#{nose}\(+|\)+#{nose}#{eyes}', "<SADFACE>", text)
    text = re.sub('#{eyes}#{nose}[\/|l*]', "<NEUTRALFACE>", text)
    text = re.sub('<3',"<HEART>", text)
    text = re.sub('[-+]?[.\d]*[\d]+[:,.\d]*', "<NUMBER>", text)
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    #text = re.sub('\[.*?\]', '', text)
    #text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation.replace('<', '').replace('>', '')), ' ', text)
    text = re.sub('\n', '', text)
    #text = re.sub('\w*\d\w*', '', text)    
    
    return text

import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
def text_preprocessing(text):
   
    tokenizer = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
    
    lemmatizer = nltk.stem.WordNetLemmatizer() 
  
    nopunc = clean_text(text)
    
    tokenized_text = tokenizer.tokenize(nopunc)
    
    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    
    lemmatized = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i,j in pos_tag(remove_stopwords)]
    
    combined_text = ' '.join(lemmatized)
    return combined_text

print(train.iloc[5680])
print(train)
train['text'] = train['text'].apply(lambda x: text_preprocessing(x))
test['text'] = test['text'].apply(lambda x: text_preprocessing(x))
print(train)


# ### Create embedding matrix

# In[300]:


import numpy as np
print(train.iloc[5680])
print(test.iloc[13])
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


# ### Retrieve embedding matrix

# In[301]:


from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train["text"])
vocab_size = len(tokenizer.word_index) + 1
#print(tokenizer.word_index)
embedding_dim = 50
embedding_matrix = create_embedding_matrix(
        '../input/glove.twitter.27B.50d.txt',
    #'../input/glove.6B.50d.txt',
    tokenizer.word_index, embedding_dim)
print(embedding_matrix)


# In[302]:


nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
nonzero_elements / vocab_size


# In[303]:


print(embedding_matrix.shape)


# ### Tokenize

# In[304]:


print(train.iloc[5680])
print(test.iloc[13])

train["text"] = tokenizer.texts_to_sequences(train["text"].values)
test["text"] = tokenizer.texts_to_sequences(test["text"].values)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(vocab_size)


# In[305]:


print(type(train.iloc[0].text))


# In[306]:


test['text'] = test['text'].apply(lambda x: [0] if x == [] else x)


# In[307]:


print(test[train[train.astype(str)['text'] == '[]']])


# ### Convert word ids to vectors

# In[308]:


'''
def tokens_to_averaged_vectors(tokens, embedding_matrix):
    vectors = np.asarray([np.asarray(embedding_matrix[token]) for token in tokens])
    return vectors.mean(axis=0)
'''


# In[309]:


'''
train_text = train["text"].apply(lambda x: tokens_to_averaged_vectors(x, embedding_matrix))
test_text = test["text"].apply(lambda x: tokens_to_averaged_vectors(x, embedding_matrix))
'''

train_text = train["text"]
test_text = test["text"]


# ### Split training and validation data

# In[310]:


from sklearn.model_selection import train_test_split

target = train["target"]
train_data, validation_data, train_target, validation_target = train_test_split(
   train_text, target, test_size=0.2, random_state=1000)
test_data = test_text


# In[311]:


validation_target.shape, train_target.shape


# ### Define Model

# In[312]:


import torch.nn as nn
import torch.nn.functional as F
def create_emb_layer(weights_matrix):
    num_embeddings, embedding_dim = weights_matrix.shape
    #emb_layer = nn.EmbeddingBag(num_embeddings, embedding_dim, sparse=True)
    emb_layer = nn.EmbeddingBag.from_pretrained(weights_matrix, freeze=True) 
    
    #if non_trainable:
        #emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class TwitterClassifier(nn.Module):
    def __init__(self, weights_matrix):
        super().__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix)
        #self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.fc = nn.Linear(embedding_dim, 2)
        self.init_weights()
        
    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded)

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

   


# ### Create custom Dataset

# In[313]:


# custom dataset
class TwitterDataset(Dataset):
    def __init__(self, texts, labels=None, transforms=None):
        self.X = texts
        if labels is not None:
            self.y = np.asarray(labels)
        else:
            self.y = None
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X.iloc[i]
        data = torch.tensor(data)
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data
        
print(train_data)
train_data = TwitterDataset(train_data, train_target)
validation_data = TwitterDataset(validation_data, validation_target)
test_data = TwitterDataset(test_data)
print(train_data[0])
print(test_data[0])


# ### Create model

# In[314]:


print(embedding_matrix.shape)


# In[315]:


model = TwitterClassifier(torch.tensor(embedding_matrix, dtype=torch.float)).to(device)
#model = TwitterClassifier(torch.tensor(embedding_matrix, dtype=torch.float)).to(device)
model


# In[316]:


def generate_batch(batch):
    label = torch.tensor([entry[1] for entry in batch])
    text = [entry[0] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


# In[317]:


from torch.utils.data import DataLoader

def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    # dataloaders
    data = DataLoader(sub_train_, shuffle=True)
    for i, (text, cls) in enumerate(data):
        optimizer.zero_grad()
        text, cls = text.to(device), cls.to(device)
        output = model(text)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def validate_func(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_)
    for text, cls in data:
        text, cls = text.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)


# ### Train the model

# In[318]:


import time
from torch.utils.data.dataset import random_split
N_EPOCHS = 10
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

sub_train_, sub_valid_ = train_data, validation_data

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = validate_func(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


# ### Predict

# In[319]:


def predict_func(test_data_):
    loss = 0
    acc = 0
    predictions = []
    data = DataLoader(test_data_)
    for text in data:
        text = text.to(device)
        with torch.no_grad():
            output = model(text)
            predictions.append(output.argmax(1))
            #acc += (output.argmax(1) == cls).sum().item()

    return predictions

predictions = predict_func(test_data)
print(len(predictions))


# ### Create submission

# In[320]:


print(predictions[0].numpy())


# In[321]:


def submission(submission_file_path,submission_data):
    sample_submission = pd.read_csv(submission_file_path)
    sample_submission["target"] = [tensor.numpy()[0] for tensor in submission_data]
    print(sample_submission["target"])
    sample_submission.to_csv("submission.csv", index=False)


# In[322]:


submission_file_path = "../input/sample_submission.csv"
submission(submission_file_path,predictions)

