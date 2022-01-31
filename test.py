#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 13:13:06 2022

@author: binger
"""

import time
import datasets, transformers, torch
from tqdm import tqdm
import pdb

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AdamW
from transformers import BertForNextSentencePrediction, AutoModel

from torch.utils.data import RandomSampler

from tools import *
from models import BertForForwardBackwardPrediction
from dataset import ddDataset
from evaluation import evaluation

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dataset = load_dataset("daily_dialog")

dialogs = dataset['train']['dialog']
dialogs_eval = dataset['validation']['dialog']
dialog_test = dataset['test']['dialog']


curr_sents, prev_sents, next_sents = constructPositives(dialogs)
curr_sents_eval, prev_sents_eval, next_sents_eval = constructPositives(dialogs_eval)
curr_sents_test, prev_sents_test, next_sents_test = constructPositives(dialogs_test)


ddtrain = constructInputs(prev_sents, curr_sents, next_sents, 'dailydialog')
ddeval = constructInputs(curr_sents_eval, prev_sents_eval, next_sents_eval, 'dailydialog')
ddtest = constructInputs(curr_sents_test, prev_sents_test, next_sents_test, 'dailydialog')


model_path = './model_checkpoints/'

fbmodel = torch.load( model_path+ 'model.epoch_3')


batch_size = 8

loader = torch.utils.data.DataLoader(ddtrain, batch_size=batch_size, shuffle=True)
loader_eval = torch.utils.data.DataLoader(ddeval, batch_size=batch_size, shuffle=True)
loader_test = torch.utils.data.DataLoader(ddtest, batch_size=batch_size, shuffle=True)


def get_dataset_acc(dataset, model, batch_size, device):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.eval()
    loop = tqdm(loader, leave=True)
    n_processed = 0
    total_correct = 0
    
    
    for batch in loop:

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
        logits = outputs.logits

        loop.set_postfix(loss=loss.item())
        n_processed += len(batch['input_ids'])*2 
        # *2 because the inputs is re-organized into 2* pairs
        # (prev, curr); (curr, next)
        
        # get num_correct from logits
        
        pred_labs = get_pred_labs(outputs.logits)
        n_correct = get_num_correct(pred_labs, labels)
        total_correct += n_correct
    print('eval loss: ', total_loss/n_processed)
    print(
        '| end of epoch {:3d} | time: {:5.2f}s |'
        'valid loss {:8.2f}'.format(
            epoch,
            (time.time() - epoch_start_time),
            total_loss/n_processed)
    )
    acc = total_correct/ n_processed
    return acc


def get_pred_labs (logits):
    pred = [1 if el[0] < el[1] else 0 for el in logits]
    return torch.LongTensor(pred)

def get_num_correct (pred_labs, true_labs):

    return (pred_labs == true_labs).float().sum()
