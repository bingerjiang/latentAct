#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 17:07:36 2022

@author: binger
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 21:21:50 2022

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
#logger = setup_logger('{}'.format('model.pt'))


#%%
dataset = load_dataset("daily_dialog")
dd_train = dataset['train']
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dialogs = dd_train['dialog']
dialogs_eval = dataset['validation']['dialog']
dialog_test = dataset['test']['dialog']

dialogs = dialogs[:20]
dialogs_eval = dialogs_eval[:20]

dialogs_flat = [utt for dialog in dialogs for utt in dialog]

bag_of_sents_tok = tokenizer(dialogs_flat, return_tensors='pt', max_length=256, truncation=True, padding='max_length')

curr_sents, prev_sents, next_sents = constructPositives(dialogs)
curr_sents_eval, prev_sents_eval, next_sents_eval = constructPositives(dialogs_eval)


ddtrain = constructInputs(prev_sents, curr_sents, next_sents, 'dailydialog')
ddeval = constructInputs(curr_sents_eval, prev_sents_eval, next_sents_eval, 'dailydialog')

#test_input_prev = tokenizer(all_prev_sents[:50], return_tensors='pt', max_length=256, truncation=True, padding='max_length')
#test_input_next = tokenizer(all_next_sents[:50], return_tensors='pt', max_length=256, truncation=True, padding='max_length')
#test_labs = torch.LongTensor(all_labs[:50]).T
#%%

####### moved to function constructInputs #######

# current_sents_tok = tokenizer(current_sents, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
# prev_sents_tok = tokenizer(prev_sents, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
# next_sents_tok = tokenizer(next_sents, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
# labels = torch.LongTensor([0]*len(current_sents)).T

# inputs = current_sents_tok

# inputs['input_ids_prev'] = prev_sents_tok['input_ids']
# inputs['input_ids_next'] = next_sents_tok['input_ids']

# inputs['token_type_ids_prev'] = prev_sents_tok['token_type_ids']
# inputs['token_type_ids_next'] = next_sents_tok['token_type_ids']


# inputs['attention_mask_prev'] = prev_sents_tok['attention_mask']
# inputs['attention_mask_next'] = next_sents_tok['attention_mask']

# inputs['labels'] = labels

# ddinput = ddDataset(inputs)
####### moved to function constructInputs #######

#%%
model = AutoModel.from_pretrained('bert-base-uncased')
# originally used BertForNextSentencePrediction
fbmodel = BertForForwardBackwardPrediction(model.config)
batch_size_train=2
batch_size_eval = 8

loader = torch.utils.data.DataLoader(ddtrain, batch_size=batch_size_train, shuffle=True)
loader_eval = torch.utils.data.DataLoader(ddeval, batch_size=batch_size_eval, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
fbmodel.to(device)
fbmodel.train()
optim = AdamW(fbmodel.parameters(), lr=5e-6)

model_path = './model_checkpoints/'
lr_decay = 5
best_eval_loss = 100000

n_plateau= 0
epochs = 2 
k =10
for epoch in range(epochs):
    
    loop = tqdm(loader, leave=True)
    total_loss = 0
    n_processed = 0
    for batch in loop:
       
        optim.zero_grad()
        
        #sample negatives
        neg_labs = [1]*k*2
        
        i = 0
        negatives = []
        while i < batch_size_train:
            sample_neg_prev_idx = RandomSampler(dialogs_flat, replacement= True, 
                                           num_samples=k)
            sample_neg_next_idx = RandomSampler(dialogs_flat, replacement= True, 
                                           num_samples=k)
            
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

            i += 1
        batch['input_ids'] = batch['input_ids'].repeat((k+1),1)
        batch['attention_mask'] = batch['attention_mask'].repeat((k+1),1)
        batch['token_type_ids'] = batch['token_type_ids'].repeat((k+1),1)

        batch['labels']=torch.cat((batch['labels'],torch.LongTensor(neg_labs).T), 0)
        
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
        outputs = fbmodel(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels)

        loss = outputs.loss
        total_loss+= loss.item()

        loss.backward()
        optim.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        n_processed += len(batch['input_ids'])*2 
        # *2 because the inputs is re-organized into 2* pairs
        # (prev, curr); (curr, next)
    print('training loss: ', total_loss/n_processed)
    
    # eval
    eval_loss = evaluation(fbmodel, loader_eval, device, epoch)
    
    #torch.save(fbmodel, model_path + '.epoch_{}'.format(epoch))
    # Save the model if the validation loss is the best we've seen so far.
    if  eval_loss < best_eval_loss:
        torch.save(fbmodel, model_path + 'model.epoch_{}'.format(epoch))
        best_eval_loss = eval_loss
        n_plateau = 0
        
    elif n_plateau >5:
        # Anneal the learning rate if no improvement has been seen in the
        # validation dataset.
        lr /= lr_decay
        n_plateau = 0
    else:
        n_plateau +=1
    
    
# test
    
#%%
# k = 10

# nsp_a, nsp_b, nsp_labs = sample_next(dialogs_flat, k)
# psp_a, psp_b, psp_labs = sample_previous(dialogs_flat, k)


# # TODO: remove duplicate rows
# all_prev_sents = nsp_a + psp_b
# all_next_sents = nsp_b + psp_a
# all_labs = nsp_labs + psp_labs