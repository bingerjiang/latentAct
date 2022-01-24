#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 22:01:37 2022

@author: binger
"""
def sample_negative_next(sents, k):
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



def sample_negative_previous(sents, k):
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

