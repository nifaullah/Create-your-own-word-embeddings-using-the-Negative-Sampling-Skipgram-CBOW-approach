# -*- coding: utf-8 -*-
"""
Created on Tue Mar 2 12:44:30 2020

@author: nifaullah
"""



import os
import numpy as np
import pandas as pd
import glob
import errno

# Path where you want to store your dataset
_path = "C:/Users/nifaullah/Downloads/msba/WinBreak/DLA/Sentiment_Analysis_RNN_IMDB/Datasets/"

# =============================================================================
# Function to create 2 dataframes each for train & test data given that the text
# files are in same structure as downloaded (i.e) ".../aclImdb/". This method
# uses another local method load one dataset which is used to create one
# dataset at a time based on the path. 
# Function doesnot take any input and returns a dictionary as output, with each
# dataframe mapped to a string.
# dict["train"] wil give you the training set & likewise for test. 
# =============================================================================
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
            
# =============================================================================
# Function to read text files from the given path and create a dataframe by
# adding each textfile as a row with it's corresponding labeled sentiment.
# function takes 2 inputs path of the file & sentiment & returns a dataframe
# with 2 columns review & sentiment
# =============================================================================

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
# Wrapper method to load IMDB dataset first it checks if a file is present in
# local already, if not then builds the dataframe & saves it to local, if
# yes loads the dataframe from the local excel file.
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