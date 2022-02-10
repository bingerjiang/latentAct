#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 22:01:37 2022

@author: binger
"""
import datasets, transformers, torch

from transformers import AutoTokenizer
from dataset import ddDataset

import logging
import pdb
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
    long_dialogs = [el for el in dataset if len(el)>2]
    
    current_sents = []
    prev_sents = []
    next_sents = []
    
    for dialog in long_dialogs:
        i = 1
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
    labels = torch.LongTensor([0]*len(curr_sents)).T
    
    inputs = current_sents_tok

    inputs['input_ids_prev'] = prev_sents_tok['input_ids']
    inputs['input_ids_next'] = next_sents_tok['input_ids']

    inputs['token_type_ids_prev'] = prev_sents_tok['token_type_ids']
    inputs['token_type_ids_next'] = next_sents_tok['token_type_ids']


    inputs['attention_mask_prev'] = prev_sents_tok['attention_mask']
    inputs['attention_mask_next'] = next_sents_tok['attention_mask']

    inputs['labels'] = labels
    
    if dataset =='dailydialog':
        initiated_inputs = ddDataset(inputs)
    
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

    inputs['labels'] = labels
    
    inputs['act'] = curr_acts
    inputs['act_prev'] = prev_acts
    inputs['act_next'] = next_acts
    
    inputs['emotion'] = curr_emotions
    inputs['emotion_prev'] = prev_emotions
    inputs['emotion_next'] = next_emotions
        
    if dataset =='dailydialog':
        initiated_inputs = ddDataset(inputs)
    
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







