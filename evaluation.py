#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 11:21:18 2022

@author: binger
"""
import datasets, transformers, torch
from tqdm import tqdm
import pdb
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AdamW
from transformers import BertForNextSentencePrediction, AutoModel

from torch.utils.data import RandomSampler

from tools import *
from models import BertForForwardBackwardPrediction
from dataset import *

@torch.no_grad()
def evaluation(model, loader_eval, device, epoch,  k, sample_negatives ):
    
    model.eval()
    loop = tqdm(loader_eval, leave=True)
    total_loss = 0
    n_processed = 0
    epoch_start_time = time.time()
    for batch in loop:

        if sample_negatives:
            neg_labs = [1]
        
            i = 0
            negatives = []
            
            sample_neg_prev_idx = RandomSampler(dialogs_flat, replacement= True, 
                                           num_samples=k*batch_size)
            sample_neg_next_idx = RandomSampler(dialogs_flat, replacement= True, 
                                           num_samples=k*batch_size)
            
            batch['input_ids_prev']=\
                    torch.cat((batch['input_ids_prev'],
                                bag_of_sents_tok['input_ids'][list(sample_neg_prev_idx)]),0)
            batch['input_ids_next']=\
                    torch.cat((batch['input_ids_next'],
                                bag_of_sents_tok['input_ids'][list(sample_neg_next_idx)]),0)
            batch['attention_mask_prev']=\
                    torch.cat((batch['attention_mask_prev'],
                                bag_of_sents_tok['attention_mask'][list(sample_neg_prev_idx)]),0)
            batch['attention_mask_next']=\
                    torch.cat((batch['attention_mask_next'],
                                bag_of_sents_tok['attention_mask'][list(sample_neg_next_idx)]),0)
            batch['token_type_ids_prev']=\
                    torch.cat((batch['token_type_ids_prev'],
                                bag_of_sents_tok['token_type_ids'][list(sample_neg_prev_idx)]),0)
            batch['token_type_ids_next']=\
                    torch.cat((batch['token_type_ids_next'],
                                bag_of_sents_tok['token_type_ids'][list(sample_neg_next_idx)]),0)

            #pdb.set_trace()    
            batch['input_ids'] = batch['input_ids'].repeat((k+1),1)
            batch['attention_mask'] = batch['attention_mask'].repeat((k+1),1)
            batch['token_type_ids'] = batch['token_type_ids'].repeat((k+1),1)
            batch['true_labels_for_cal_acc'] = torch.cat((batch['labels'],torch.LongTensor(neg_labs).T.repeat(k*batch_size)), 0).repeat(2)

            batch['labels']=torch.cat((batch['labels'],torch.LongTensor(neg_labs).T.repeat(k*batch_size)), 0)
        
       
        # positive samples
        input_ids = [batch['input_ids_prev'].to(device),
                     batch['input_ids'].to(device),
                     batch['input_ids_next'].to(device)]
        attention_mask = [batch['attention_mask_prev'].to(device),
                          batch['attention_mask'].to(device),
                          batch['attention_mask_next'].to(device)]
        token_type_ids = [batch['token_type_ids_prev'].to(device),
                          batch['token_type_ids'].to(device),
                          batch['token_type_ids_next'].to(device)]
        labels = batch['labels'].to(device)
        
        #pdb.set_trace()
        outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels)

        loss = outputs.loss
        total_loss+= loss.item()

        #loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        n_processed += len(batch['input_ids'])*2 
        # *2 because the inputs is re-organized into 2* pairs
        # (prev, curr); (curr, next)
    print('eval loss: ', total_loss/n_processed)
    print(
        '| end of epoch {:3d} | time: {:5.2f}s |'
        'valid loss {:8.2f}'.format(
            epoch,
            (time.time() - epoch_start_time),
            total_loss/n_processed)
    )
    return total_loss/n_processed














