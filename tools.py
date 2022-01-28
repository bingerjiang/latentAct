#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 22:01:37 2022

@author: binger
"""
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













