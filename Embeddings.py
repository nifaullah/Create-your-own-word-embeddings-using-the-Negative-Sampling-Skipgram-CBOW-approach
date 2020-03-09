# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:35:43 2020

@author: nifaullah
"""

import os
import numpy as np
import pandas as pd
import glob
import errno
import keras
from keras.layers import Input, Embedding, Dense, Flatten
from keras.models import Model, Sequential
import re
from random import sample

_path = "C:/Users/nifaullah/Downloads/msba/WinBreak/DLA/Sentiment_Analysis_RNN_IMDB/Datasets/"
_unk = "<UNK>"
_eos = "<EOS>"

def LoadDatasets():
    dataset_dict = {}
    dir = "C:/Users/nifaullah/Downloads/msba/WinBreak/DLA/IMDB_RNN/aclImdb/"
    env = ["train", "test"]
    sentiment = ["pos", "neg"]
    all_text = "*.txt"
    for _env in env:
        df = pd.DataFrame()
        for _sentiment in sentiment:
            path = f"{dir}{_env}/{_sentiment}/{all_text}"
            df = pd.concat([df,LoadOneDataset(path, _sentiment)])
        dataset_dict[_env] = df
    return dataset_dict
            
def LoadOneDataset(path, sentiment):
    sentiment_dict = {"neg": 0, "pos": 1}
    files = glob.glob(path)
    content = []
    
    with open(name, 'r', encoding="utf8") as file:
        content.append(file.readlines())
    
    df = pd.DataFrame(content,columns=["review"])
    df["sentiment"] = sentiment_dict[sentiment]
    return df

# =============================================================================
# Load data using the user created Load_IMDB_Datasets library if data is 
# already not present as an excel in local. If you 're running this for first
# time This will take sometime(15 -30 mins) depending on your computer memory.
# =============================================================================
def LoadImdbDatasets():
    _train = "train.xlsx"
    _test = "test.xlsx"
    if not (os.path.isfile(f"{_path}{_train}") & os.path.isfile(f"{_path}{_test}")):
        datasets = imdb.LoadDatasets()
        train_df = datasets["train"]
        test_df = datasets["test"]
        train_df.to_excel(f"{path}{_train}", index=False)
        test_df.to_excel(f"{path}{_test}", index=False)
    else:
        train_df = pd.read_excel(f"{_path}{_train}")
        test_df = pd.read_excel(f"{_path}{_test}")
    return train_df, test_df

def CleanDataset(df):
    df = df.copy()
    #lower 
    df.review = df.review.str.lower()
    #Removing html tags
    df.review = df.review.apply(lambda x: re.compile(r'<[^>]+>').sub('', str(x)))
    #Removing Punctuations
    df.review = df.review.str.replace('[^a-zA-Z ]', '')
    return df

def GetWord2Index(df, max_occurences=30):
    word_count = pd.Series([y for x in df.values.flatten() for y in str(x).split()]).value_counts()
    selected_words = word_count[word_count > max_occurences]
    vocab = {x:index for index, x in enumerate(selected_words.index)}
    vocab.update({_eos: len(vocab), _unk: len(vocab)+1})
    return vocab

def CreateCorpus(df, name=""):
    _corpusfile = "corpus.txt"
   # only_vocab_words_regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, deleted_words)))
    if not (os.path.isfile(f"{_path}{name}_{_corpusfile}")):
        corpus = ""
        #counter = 0
        for i in df.values:
            corpus = f"{corpus}{i[0]} {_eos} "
        textfile = open(f"{_path}{name}_{_corpusfile}", "w")
        textfile.write(corpus)
        textfile.close()
    else:
        with open(f"{_path}{name}_{_corpusfile}", 'r', encoding="utf8") as file:
            corpus = file.read()
    return corpus

def GetWindow(current_index, max_len, window_size = 3):
    lower = max(0, current_index - window_size)
    upper = min(current_index + window_size, max_len-1) + 1
    return lower,upper

def CreateSkipgramContextTargetPairs(corpus, vocab, window_size = 3):
    tokens = corpus.split()
    x = np.zeros((len(tokens),1))
    rows_to_remove = 0
    no_of_outputs = window_size*2
    y = np.tile(len(vocab),len(tokens)*no_of_outputs)
    y = np.reshape(y, (len(tokens), no_of_outputs))
    
    max_len = len(tokens)
    for i in range(max_len):        
        column_no = 0
        if tokens[i] in vocab:
            lower,upper = GetWindow(i,max_len, window_size)
            x[i,0] = vocab[tokens[i]]
            for j in range(lower,upper):
                if tokens[j] in vocab and i != j:
                    y[i,column_no] = vocab[tokens[j]]
                    column_no += 1                    
        else:
            rows_to_remove += 1
    rows_to_remove =  max_len - rows_to_remove
    x = x[:rows_to_remove,]
    y = y[:rows_to_remove,]
    return x, y

def CreateNegativeSamplingContextTargetPairs(corpus, vocab, window_size = 3, neg_samples = 5):
    _data = "embeddings_data.npy"
    _labels = "embeddings_labels.npy"
    if os.path.isfile(f"{_path}{_data}") and os.path.isfile(f"{_path}{_labels}"):
        print("yay")
        X = np.load(f"{_path}{_data}", allow_pickle = True)
        Y = np.load(f"{_path}{_labels}", allow_pickle = True)
        return X, Y
        
    tokens = corpus.split()
    cols = []
    match = []
    max_len = len(tokens)
    neg_samples_dict = {}
    
    for i in range(max_len):
        print(max_len-i)
        if tokens[i] in vocab:
            if tokens[i] not in neg_samples_dict.keys():
                neg_samples_dict[vocab[tokens[i]]] = list(vocab.values())
            lower,upper = GetWindow(i,max_len, window_size)
            for j in range(lower,upper):
                if tokens[j] in vocab and i != j:
                    cols.append(np.array([vocab[tokens[i]], vocab[tokens[j]]], dtype=int))
                    match.append(1)
                    if vocab[tokens[j]] in neg_samples_dict[vocab[tokens[i]]]:
                        neg_samples_dict[vocab[tokens[i]]].remove(vocab[tokens[j]])
    
    pos_data_length = len(match)
    print(pos_data_length)
    for i in range(pos_data_length):
        print(pos_data_length-i)
        #print(cols[i][0])
        negative_targets = sample(neg_samples_dict[cols[i][0]], neg_samples)
        #print(negative_targets)
        match.extend(np.repeat(0, neg_samples))
        cols.extend(np.column_stack((np.repeat(cols[i][0], neg_samples), negative_targets)))
    
    neg_data_length = len(match) - pos_data_length
    print(pos_data_length)
    print(neg_data_length)
    
    X = np.array(cols)
    Y = np.array(match)
    
    np.save(f"{_path}{_data}",X)
    np.save(f"{_path}{_labels}",Y)
    
    return X, Y

def BuildModel(vocab_size, emb_size, window_size):
    print(vocab_size)
    model = Sequential([
        Flatten(input_shape=(1,2)),
        #Embedding(output_dim=emb_size, input_dim=vocab_size),
        Dense(1)])
    return model
    #main_input = Input(shape=(1,), dtype='int32', name='main_input')
    #X = Embedding(output_dim=emb_size, input_dim=vocab_size, input_length = 1)(main_input)
    #X = Flatten()(X)
    #dense1 = Dense(vocab_size)(X)
    #   x1 = keras.layers.concatenate([X, dense1])
    # dense2 = Dense(vocab_size)(X)
    # x2 = keras.layers.concatenate([X, dense2])
    # dense3 = Dense(vocab_size)(X)
    # x3 = keras.layers.concatenate([X, dense3])
    # dense4  = Dense(vocab_size)(X)
    # x4 = keras.layers.concatenate([X, dense4])
    # dense5  = Dense(vocab_size)(X)
    # x5 = keras.layers.concatenate([X, dense5])
    # dense6  = Dense(vocab_size)(X)
    # x6 = keras.layers.concatenate([X, dense6])
    # model = Model(inputs=main_input, outputs=[dense1, dense2, dense3, dense4, dense5, dense6])
    # model = Model(inputs=main_input, outputs=dense1)
    #return model

def TrainModel(X_train, Y_train, vocab_size, emb_size = 300, window_size = 3, epochs = 1, optimizer = 'adam'):
    labels = []
    print(X_train.shape)
    print(Y_train.shape)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(2,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2)])
    model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(X_train, Y_train)
    
def test():
    train_df, test_df = LoadImdbDatasets()
    #train_df = CleanDataset(train_df)
    #test_df = CleanDataset(test_df)
    return train_df

def BuildEmbeddings(df, max_occurences = 30, emb_size = 300, window_size = 3, epochs = 10, optimizer = 'adam'):
    df = CleanDataset(df)
    word2index_vocab = GetWord2Index(df,max_occurences)
    index2word_vocab = dict(zip(word2index_vocab.values(), word2index_vocab.keys()))
    corpus = CreateCorpus(train_df[["review"]],"imdb")
    X,Y =  CreateNegativeSamplingContextTargetPairs(corpus, word2index_vocab, window_size, 1)
    #X_train = np.reshape(X_train, (X_train.shape[0],1,2))
    #TrainModel(X_train, Y_train, len(word2index_vocab), emb_size, epochs, optimizer)
    return X, Y, word2index_vocab, index2word_vocab
    
train_df = test()

#vocab = GetWord2Index(train_df, 30)
#print(len(vocab))  
#corpus = CreateCorpus(train_df[["review"]],"imdb")

#X,Y, r = CreateContextTargetPairs(corpus, vocab)

#model = BuildModel(len(vocab), 300, 3)   


X, Y , word2index_vocab, index2word_vocab = BuildEmbeddings(train_df)
#word2index_vocab = GetWord2Index(train_df,30)
#TrainModel(X, Y, len(word2index_vocab), 300, 1, 'adam')
# k = X.shape[0]
# x = np.zeros((k,2))
# for i in range(k):
#     print(k - i)
#     x[i] = X[i].astype(int)