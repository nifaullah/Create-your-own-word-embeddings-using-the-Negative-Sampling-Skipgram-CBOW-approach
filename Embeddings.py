# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:35:43 2020
@author: nifaullah
"""

import os
import numpy as np
import pandas as pd
import IMDB_utils 
import keras
from keras.layers import Input, Embedding, Dense, Flatten
from keras.models import Model, Sequential, load_model
from random import sample
import tensorflow as tf
import warnings

# To ignore Keras pre-emptive sparse vector warning
# Issue mentioned here - https://github.com/matterport/Mask_RCNN/issues/749
warnings.filterwarnings('ignore')
# Path where you want to store your dataset
_path = "C:/Users/nifaullah/Downloads/msba/WinBreak/DLA/Sentiment_Analysis_RNN_IMDB/Datasets/"
# Token for words not in dictionary
_unk = "<UNK>"
# Token to recognize end of sentence
_eos = "<EOS>"

# =============================================================================
# Function to clean dataset before processing. Here we lower case the review
# column and remove HTML tags and punctuations. Ideally you don't want
# prepositions or articles to be part of word vector but I will adress this in
# future iterations.
# Function takes dataframe as an input and returns a dataframe.
# Input:
#   1. df - Raw Dataframe
# Output:
#   1. Cleaned Dataframe
# =============================================================================
def CleanDataset(df):
    df = df.copy()
    #lower 
    df.review = df.review.str.lower()
    #Removing html tags
    df.review = df.review.apply(lambda x: re.compile(r'<[^>]+>').sub('', str(x)))
    #Removing Punctuations
    df.review = df.review.str.replace('[^a-zA-Z ]', '')
    return df

# =============================================================================
# Function to generate the vocabulary
# Takes dataframe as an input along with max_occurences (i.e max number of
# times a word should appear before it can be included in the vocabulary) and
# returns a dictionary with each word (key) mapped to an index (value).
# Input:
#   1. df - dataframe from which the vocabulary is to be generated
#   2. max_occurences - max number of times a word should appear before it can 
#       be included in the vocabulary
# Output:
#   1. Dictionary with word as key and index as value
# =============================================================================
def GetWord2Index(df, max_occurences=30):
    word_count = pd.Series([y for x in df.values.flatten() for y in str(x).split()]).value_counts()
    selected_words = word_count[word_count > max_occurences]
    word2index_vocab = {x:index for index, x in enumerate(selected_words.index)}
    word2index_vocab.update({_eos: len(vocab), _unk: len(vocab)+1})
    return word2index_vocab

# =============================================================================
# Funtcion to reverse the dictionary with now index as key and word as value
# Input:
#    1. Dictionary with word as key and index as value
# Output:
#    1. Dictionary with index as key and word as value
# =============================================================================
def GetIndex2Word(word2index_vocab):
    index2word_vocab = dict(zip(word2index_vocab.values(), word2index_vocab.keys()))
    return index2word_vocab


# =============================================================================
# Function to create a locally contextualized corpus
# Takes dataframe as an input - ideally this a column which contains all the 
# text (for instance for the IMDB dataframe this will be review column typed
# as dataframe) and another input is the name of the corpus so that the corpus
# is saved locally and we don't have to create a corpus eachtime.
# This function returns the corpus as a string.
# Input:
#   1. df - dataframe from which the corpus is to be generated
#   3. name - string containing the prefix of the corpus to be retrieved
#       from the local or to be saved to the local
# Outputs:
#   1. string containing entire corpus
# =============================================================================
def CreateCorpus(df, name=""):
    _corpusfile = "corpus.txt"
    if not (os.path.isfile(f"{_path}{name}_{_corpusfile}")):
        corpus = ""
        for i in df.values:
            corpus = f"{corpus}{i[0]} {_eos} "
        textfile = open(f"{_path}{name}_{_corpusfile}", "w")
        textfile.write(corpus)
        textfile.close()
    else:
        with open(f"{_path}{name}_{_corpusfile}", 'r', encoding="utf8") as file:
            corpus = file.read()
    return corpus

# =============================================================================
# Function to get the index of contextual words around a word to create context
# target pairs. This takes the current_index (integer representing the current
# word), max_len (integer indicating the max number of words in the corpus)
# and window_size (integer which defines the context word around the target
# word) and returns an integer tuple with lower bound and upper bound for the
# context window.
# Input:
#   1. current_index - integer containing the index of current target word
#   2. max_len  - integer containing number of tokens available
#   3. window_size - integer to get neighbouring words indices based on the
#        window
# Outputs:
#   1. Lower bound for contextual neighbours of a particular target word
#   2. Upper bound for contextual neighbours of a particular target word
# =============================================================================
def GetWindow(current_index, max_len, window_size = 3):
    lower = max(0, current_index - window_size)
    upper = min(current_index + window_size, max_len-1) + 1
    return lower,upper

# =============================================================================
# Function to create training dataset to be fed in the neural network,following
# the original paper input is the target word or centered word whilst the output
# is the neighbouring words based on the window size. Input is a vector whereas
# output is a matrix of size (token_size, window size * 2). Because of the huge
# output size, counting in the need for one hot encoding each of those outputs,
# the number of softmax regression to be done makes the operation quite complex,
# as a result of which I wasn't able to test the output of this method on the
# neural network, and I had to resort to negative sampling which is also
# mentioned in the original paper. If one wants to use Continuos Bag Of Words
# (CBOW) appraoch to build embeddings they can still use the method but the
# labels & data would interchange.
# Inputs:
#   1. corpus - String containing entire corpus
#   2. vocab - Dictionary with Word to Index mapping
#   3. window_size - integer to define number of context words for each target
# Outputs:
#   1. X - training data with shape (# of valid tokens, 1)
#   2. Y - training label with shape (# of valid tokens, window_size * 2)
# =============================================================================
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

# =============================================================================
# Function to create context target pairs along with negative samples,following
# the original paper inputs are the target word and context word, neighbouring
# words based on the window size. Input is a matrix containing n datapoints
# with 2 features and the label/output is a vector containing binary values
# 1 if context word is a neigbour & 0 if it's negatively sampled.
#    
# WARNING !!: This method will take huge amount of time to run if you're
# running it in local PC. For my PC with 8GB RAM, to generate a dataset of
# nearly 60 Million samples with window_size = 3 & negative_samples = 1 it took
# close to 23 hours. After which I decided to save the  training set locally
# and use it if it's available.
# Inputs:
#   1. corpus - string containing entire corpus
#   2. vocab - dictionary with Word to Index mapping
#   3. window_size - integer to define number of context words for each target
#   4. neg_samples - integer to generate number of negative samples per
#       positive sample
# Outputs:
#   1. X - training data with shape (# of valid tokens, 2)
#   2. Y - training label with shape (# of valid tokens,)
# =============================================================================
def CreateNegativeSamplingContextTargetPairs(corpus, vocab, window_size = 3, neg_samples = 5):
    _data = "embeddings_data.npy"
    _labels = "embeddings_labels.npy"
    if os.path.isfile(f"{_path}{_data}") and os.path.isfile(f"{_path}{_labels}"):
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
    print("Positive Samples completed")
    for i in range(pos_data_length):
        print(pos_data_length-i)
        negative_targets = sample(neg_samples_dict[cols[i][0]], neg_samples)
        match.extend(np.repeat(0, neg_samples))
        cols.extend(np.column_stack((np.repeat(cols[i][0], neg_samples), negative_targets)))
    
    print("Negative Samples completed")
    X = np.array(cols)
    Y = np.array(match)
    
    np.save(f"{_path}{_data}",X)
    np.save(f"{_path}{_labels}",Y)
    
    return X, Y

# =============================================================================
# Function to build sequential model, this is essentially a basic logistic
# regression model preceded by an embedding layer
# Inputs:
#   1. vocab_size - integer containing the length of the vocabulary
#   2. emb_size - integer defining the dimension for the word vector
# Output:
#   1. Sequential Logistic Regression model preceded by an embedding layer
# =============================================================================
def BuildModel(vocab_size, emb_size):
    model = Sequential([
        Embedding(output_dim=emb_size, input_dim=vocab_size, input_length = 2),
        Flatten(input_shape=(1,2)),
        Dense(2)])
    return model

# =============================================================================
# Function of train the model, although I have given access to hyperparameters
# such as optimizer & epochs but internally epoch is still hardcoded & this
# along with batch_size will be accomadated in coming iterations.
# Inputs:
#   1. X_train - Numpy integer array with shape (# samples,2)
#   2. Y_train - Numpy integer array with shape (# samples,)
#   3. emb_size - integer defining the dimension for the word vector
#   4. window_size - integer to define number of context words for each target
#   5. epochs - integer defining # of epochs to train the model for.
#   6. optimizer - string defining the optimizer to be used for training
# Output:
#   1. A trained model
# =============================================================================
def TrainModel(X_train, Y_train, vocab_size, emb_size = 300, window_size = 3, epochs = 1, optimizer = 'adam'):
    labels = []
    print(X_train.shape)
    print(Y_train.shape)
    model = BuildModel(vocab_size, emb_size, window_size)
    model.compile(optimizer= optimizer,
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=1, batch_size=128)
    return model

# =============================================================================
# Parent Function which will be exposed to clients who want to create their own
# locally contextualized web embeddings. It takes the dataframe and many other
# hyperparameters as inputs and returns the word vectors.
# Inputs:
#   1. df - dataframe containing only the text to be read as corpus.
#   2. force_train - boolean, if true train the model, if not load from local
#   3. max_occurences - max number of times a word should appear before it can 
#       be included in the vocabulary
#   4. emb_size - integer defining the dimension for the word vector
#   5. window_size - integer to define number of context words for each target
#   6. corpus_name - string containing the prefix of the corpus to be retrieved
#       from the local or to be saved to the local
#   7. epochs - integer defining # of epochs to train the model for.
#   8. optimizer - string defining the optimizer to be used for training
# Output:
#   1. A numpy float array with shape (vocab_size, emb_size) containing the
#       vectors for each word 
# =============================================================================
def BuildEmbeddings(df, force_train = False,max_occurences = 30, emb_size = 300, window_size = 3, corpus_name = "", epochs = 1, optimizer = 'adam'):
    _model = "imdb_negative_sampling_model.h5"
    if (os.path.isfile(f"{_path}{_model}") and not force_train):
        model = load_model(f"{_path}{_model}")
        return model.layers[0].get_weights()[0]
    df = CleanDataset(df)
    word2index_vocab = GetWord2Index(df,max_occurences)
    index2word_vocab = GetIndex2Word(word2index_vocab)
    corpus = CreateCorpus(df,corpus_name)
    X,Y =  CreateNegativeSamplingContextTargetPairs(corpus, word2index_vocab, window_size, 1)
    model = TrainModel(X, Y, len(word2index_vocab), emb_size, epochs, optimizer)
    model.save(f"{_path}{_model}")
    return model.layers[0].get_weights()[0]


# =============================================================================
# A sample test function to sanity check the output and also demonstrates
# how to use the parent function minimally.
# Output:
#    1. returns word_vector
# =============================================================================
def TestEmbeddings():
    train_df, test_df = IMDB_utils.LoadImdbDatasets()
    word_vector = BuildEmbeddings(train_df[["review"]], corpus_name = "imdb")
    return word_vector

# Calling the test function to check the sanity of the code
#word_vectors = TestEmbeddings()
