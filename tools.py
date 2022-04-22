#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 22:01:37 2022

@author: binger
"""
import datasets, transformers, torch

from transformers import AutoTokenizer
from dataset import *


import pdb
import datetime
import logging
import os
def sample_next(sents, k):
    '''
    Parameters
    ----------
    sents : list
        list of utterances in dialogues
    k : int
        negative:positive

    Returns
    -------
    sents_a : TYPE
        DESCRIPTION.
    sents_b : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.
    '''
    
    sents_a = []
    sents_b = []
    labels = []
    
    i = 1+k
    while i < len(sents)-1-k:
        current_sent = [sents[i]]
        current_sents = current_sent*(k+1)
        #next_sents = [sents[i+1]]
        
        next_sents = [sents[i+1+j] for j in range(k+1)]
        
        label = [0] + [1]*k
        
        sents_a += current_sents
        sents_b += next_sents
        labels += label
        
        i +=1

    return sents_a, sents_b, labels



def sample_previous(sents, k):
    '''
    Parameters
    ----------
    sents : list
        list of utterances in dialogues
    k : int
        negative:positive

    Returns
    -------
    sents_a : TYPE
        DESCRIPTION.
    sents_b : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.
    '''
    
    sents_a = []
    sents_b = []
    labels = []
    
    i = 1+k
    while i <len(sents)-1-k:
        current_sent = [sents[i]]
        current_sents = current_sent*(k+1)
        #next_sents = [sents[i+1]]
        
        prev_sents = [sents[i-1-j] for j in range(k+1)]
        
        label = [0] + [1]*k
        
        sents_a += current_sents
        sents_b += prev_sents 
        labels += label
        
        i +=1

    return sents_a, sents_b, labels

#%%

def constructPositives (dataset):
    '''
    dataset: list of lists
    current_sents: not the first of last sent in dialog
    '''
    # exclude dialogs that only have 2 utterances
    long_dialogs = [el for el in dataset if len(el['turns'])>2]
    
    current_sents = []
    prev_sents = []
    next_sents = []
    #pdb.set_trace()

    for dia in long_dialogs:
        i = 1
        dialog = dia['turns']
        while i < len(dialog)-1:
            current_sents.append(dialog[i])
            prev_sents.append(dialog[i-1])
            next_sents.append(dialog[i+1])
           
            i +=1
   
    return current_sents, prev_sents, next_sents

def constructPositives_dataset (dataset):
    '''
    dataset: list of lists
    current_sents: not the first of last sent in dialog
    '''
    # exclude dialogs that only have 2 utterances
    long_dialogs = [el for el in dataset if len(el['dialog'])>2]
    
    #long_dialogs = long_dialogs_dataset['dialog']
    current_sents = []
    prev_sents = []
    next_sents = []
    #pdb.set_trace()
    
    curr_acts = []
    prev_acts = []
    next_acts = []
    
    curr_emotions = []
    prev_emotions = []
    next_emotions = []
    for dia in long_dialogs:
        #dialog is dict
        i = 1
        dialog = dia['dialog']
        act = dia['act']
        emotion = dia['emotion']
        
        while i < len(dialog)-1:
            current_sents.append(dialog[i])
            prev_sents.append(dialog[i-1])
            next_sents.append(dialog[i+1])
            
            curr_acts.append(act[i])
            prev_acts.append(act[i-1])
            next_acts.append(act[i+1])
            
            curr_emotions.append(emotion[i])
            prev_emotions.append(emotion[i-1])
            next_emotions.append(emotion[i+1])           
            i +=1
    print('len of current_sents: ',len(current_sents))
    sents = [current_sents, prev_sents, next_sents]
    acts = [curr_acts, prev_acts, next_acts]
    emotions = [curr_emotions, prev_emotions, next_emotions]
    return sents, acts, emotions
#%%
def constructInputs (prev_sents, curr_sents, next_sents, dataset):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    current_sents_tok = tokenizer(curr_sents, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    prev_sents_tok = tokenizer(prev_sents, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    next_sents_tok = tokenizer(next_sents, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    
    
    inputs = current_sents_tok
    
    inputs['input_ids_prev'] = prev_sents_tok['input_ids']
    inputs['input_ids_next'] = next_sents_tok['input_ids']

    inputs['token_type_ids_prev'] = prev_sents_tok['token_type_ids']
    inputs['token_type_ids_next'] = next_sents_tok['token_type_ids']


    inputs['attention_mask_prev'] = prev_sents_tok['attention_mask']
    inputs['attention_mask_next'] = next_sents_tok['attention_mask']
    labels = torch.LongTensor([0]*len(curr_sents)).T
    inputs['labels'] = labels
    
        
    
    initiated_inputs = initializeDataset(inputs)
    
    return initiated_inputs

def constructInputs_with_act_emotion (sents, acts, emotions, dataset):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #pdb.set_trace()
    curr_sents, prev_sents, next_sents = sents
    curr_acts, prev_acts, next_acts = acts
    curr_emotions, prev_emotions, next_emotions = emotions
    
    current_sents_tok = tokenizer(curr_sents, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    prev_sents_tok = tokenizer(prev_sents, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    next_sents_tok = tokenizer(next_sents, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    labels = torch.LongTensor([0]*len(curr_sents)).T
    
    inputs = current_sents_tok

    inputs['input_ids_prev'] = prev_sents_tok['input_ids']
    inputs['input_ids_next'] = next_sents_tok['input_ids']

    inputs['token_type_ids_prev'] = prev_sents_tok['token_type_ids']
    inputs['token_type_ids_next'] = next_sents_tok['token_type_ids']


    inputs['attention_mask_prev'] = prev_sents_tok['attention_mask']
    inputs['attention_mask_next'] = next_sents_tok['attention_mask']

    if dataset =='daily_dialog':
        inputs['labels'] = labels
        
        inputs['act'] = curr_acts
        inputs['act_prev'] = prev_acts
        inputs['act_next'] = next_acts
        
        inputs['emotion'] = curr_emotions
        inputs['emotion_prev'] = prev_emotions
        inputs['emotion_next'] = next_emotions
        
    
    
    initiated_inputs = initializeDataset(inputs)
    #pdb.set_trace()
    #initiated_inputs.set_format('torch', columns=['prev_sents', 'curr_sents','next_sents'])
    #initiated_inputs.set_format("torch", column=["curr_sents"])
    #initiated_inputs.set_format("torch", column=["next_sents"])
    return initiated_inputs



def setup_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    # create file handler which logs even debug messages
    fh = logging.FileHandler('log/%s.log' % logger_name)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def initiateTokenizedInputs (sents,embed_type, dataset='meta_woz'):
    if embed_type == 'bert':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif embed_type == 'sbert':
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    elif embed_type == 'simcse':
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    sents_tok = tokenizer(sents, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    initiated_inputs = initializeDataset(sents_tok)
    
    return initiated_inputs





def setup_logs(save_dir, run_name='none'):
    ## copy from cpc-nlp-pytorch
    
    if run_name =='none':
        now = datetime.datetime.now()
        curr_date = now.strftime("%Y-%m-%d_%H_%M_%S")
        run_name = curr_date
    # initialize logger
    logger = logging.getLogger("cpc")
    logger.setLevel(logging.INFO)

    # create the logging file handler
    log_file = os.path.join(save_dir, run_name + ".log")
    fh = logging.FileHandler(log_file)

    # create the logging console handler
    ch = logging.StreamHandler()

    # format
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)

    # add handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


