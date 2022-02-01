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


#%%
def get_dataset_acc(dataset, model, batch_size, device, sample_negatives):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.eval()
    loop = tqdm(loader, leave=True)
    n_processed = 0
    total_correct = 0
    total_loss = 0
    k=1
    data_rep = dict()
    data_rep['true_labels'] = torch.LongTensor().to(device)
    data_rep['pred_labels'] = torch.LongTensor().to(device)
    data_rep['prev_forward'] = torch.Tensor().to(device)
    data_rep['curr_forward'] = torch.Tensor().to(device)
    data_rep['curr_backward'] = torch.Tensor().to(device)
    data_rep['next_backward'] = torch.Tensor().to(device)
    data_rep['input_ids_prev'] = torch.Tensor().to(device)
    data_rep['input_ids'] = torch.Tensor().to(device)
    data_rep['input_ids_next'] = torch.Tensor().to(device)
    
    for batch in loop:

        ############### negative samples ##########
        if len(batch['input_ids_prev']) != batch_size:
            break
        if sample_negatives:
            neg_labs = [1]*k
        
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
            ## double len of labels for cal acc
            ## because two pairs for each input row
        ############### end negative samples ##########
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
        
        pred_labs = get_pred_labs(outputs.logits).to(device)
        #import pdb; pdb.set_trace()
        #labels = batch['true_labels_for_cal_acc'].to(device)
        #pdb.set_trace()
        if sample_negatives:
            labels = batch['true_labels_for_cal_acc'].to(device)
        n_correct = get_num_correct(pred_labs, labels)
        total_correct += n_correct

        ####### add to return data ######
        data_rep['true_labels'] = torch.cat((data_rep['true_labels'], labels), 0)
        data_rep['pred_labels'] = torch.cat((data_rep['pred_labels'], pred_labs), 0)
        # [prev_forward, curr_forward, curr_backward, next_backward]
        data_rep['prev_forward'] = torch.cat((data_rep['prev_forward'], outputs.hidden_states[0]),0)
        data_rep['curr_forward'] = torch.cat((data_rep['curr_forward'], outputs.hidden_states[1]),0)
        data_rep['curr_backward'] = torch.cat((data_rep['curr_backward'], outputs.hidden_states[2]),0)
        data_rep['next_backward'] = torch.cat((data_rep['next_backward'], outputs.hidden_states[3]),0)
        #data_rep['input_ids_prev'] = torch.cat((data_rep['input_ids_prev'], batch['input_ids_prev'].to(device)),0)
        #data_rep['input_ids'] = torch.cat((data_rep['input_ids'], batch['input_ids'].to(device)),0)
        #data_rep['input_ids_next'] = torch.cat((data_rep['input_ids_next'], batch['input_ids_next'].to(device)),0)

        #################################
    print('eval loss: ', total_loss/n_processed)

    acc = total_correct/ n_processed
    #pdb.set_trace()
    return acc, data_rep


def get_pred_labs (logits):
    pred = [1 if el[0] < el[1] else 0 for el in logits]
    return torch.LongTensor(pred)

def get_num_correct (pred_labs, true_labs):

    return (pred_labs == true_labs).float().sum()



#%%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dataset = load_dataset("daily_dialog")

dialogs = dataset['train']['dialog']
dialogs_eval = dataset['validation']['dialog']
dialogs_test = dataset['test']['dialog']


dialogs_flat = [utt for dialog in dialogs for utt in dialog]

bag_of_sents_tok = tokenizer(dialogs_flat, return_tensors='pt', max_length=128, truncation=True, padding='max_length')

curr_sents, prev_sents, next_sents = constructPositives(dialogs)
curr_sents_eval, prev_sents_eval, next_sents_eval = constructPositives(dialogs_eval)
curr_sents_test, prev_sents_test, next_sents_test = constructPositives(dialogs_test)


ddtrain = constructInputs(prev_sents, curr_sents, next_sents, 'dailydialog')
ddeval = constructInputs(curr_sents_eval, prev_sents_eval, next_sents_eval, 'dailydialog')
ddtest = constructInputs(curr_sents_test, prev_sents_test, next_sents_test, 'dailydialog')


model_path = './old/model_checkpoints/'



model = AutoModel.from_pretrained('bert-base-uncased')
# originally used BertForNextSentencePrediction
bertmodel = BertForForwardBackwardPrediction(model.config)

fbmodel = torch.load( model_path+ 'lr=1e-5_model.epoch_3')

# lr=1e-5_model.epoch_3
# 0.9793
# lr=5e-6_model.epoch_3
# 0.9679

batch_size = 3

loader = torch.utils.data.DataLoader(ddtrain, batch_size=batch_size, shuffle=True)
loader_eval = torch.utils.data.DataLoader(ddeval, batch_size=batch_size, shuffle=True)
loader_test = torch.utils.data.DataLoader(ddtest, batch_size=batch_size, shuffle=True)



#acc_test = get_dataset_acc(ddtest, bertmodel, batch_size, device, True)
#print('bert test acc: ',acc_test)

#acc_eval, fb_eval_rep = get_dataset_acc(ddeval, fbmodel, batch_size, device, True)
#print('eval acc: ',acc_eval)

acc_test, fb_test_rep = get_dataset_acc(ddtest, fbmodel, batch_size, device, True)
print('fbmodel test acc: ',acc_test)
pdb.set_trace()
print('done')


#acc_train = get_dataset_acc(ddtrain, fbmodel, batch_size, device)
#print('train acc: ',acc_train)


