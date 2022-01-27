#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 21:21:50 2022

@author: binger
"""


import datasets, transformers, torch
from tqdm import tqdm
import pdb

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AdamW
from transformers import BertForNextSentencePrediction, AutoModel



from tools import *
from models import BertForForwardBackwardPrediction
from dataset import ddDataset


dataset = load_dataset("daily_dialog")
dd_train = dataset['train']
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dialogs = dd_train['dialog']

dialogs = [[' '] + el  for el in dialogs]


dialogs_flat = [utt for dialog in dialogs for utt in dialog]
dialogs_flat += [' ']

k = 10

nsp_a, nsp_b, nsp_labs = sample_next(dialogs_flat, k)
psp_a, psp_b, psp_labs = sample_previous(dialogs_flat, k)


# TODO: remove duplicate rows
all_prev_sents = nsp_a + psp_b
all_next_sents = nsp_b + psp_a
all_labs = nsp_labs + psp_labs


test_input_prev = tokenizer(all_prev_sents[:50], return_tensors='pt', max_length=256, truncation=True, padding='max_length')
test_input_next = tokenizer(all_next_sents[:50], return_tensors='pt', max_length=256, truncation=True, padding='max_length')
test_labs = torch.LongTensor(all_labs[:50]).T
#%%
test_inputs = test_input_prev
test_inputs['labels'] = test_labs
test_inputs['input_ids2'] = test_input_next['input_ids']
test_inputs['token_type_ids2'] = test_input_next['token_type_ids']
test_inputs['attention_mask2'] = test_input_next['attention_mask']


testdd = ddDataset(test_inputs)


model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
fbmodel = BertForForwardBackwardPrediction(model.config)
loader = torch.utils.data.DataLoader(testdd, batch_size=16, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
fbmodel.to(device)
fbmodel.train()
optim = AdamW(model.parameters(), lr=5e-6)

epochs = 2

for epoch in range(epochs):
    
    loop = tqdm(loader, leave=True)
    for batch in loop:
       
        optim.zero_grad()
       
        input_ids = [batch['input_ids'].to(device),batch['input_ids2'].to(device)]
        attention_mask = [batch['attention_mask'].to(device),batch['attention_mask2'].to(device)]
        token_type_ids = [batch['token_type_ids'].to(device),batch['token_type_ids2'].to(device)]
        labels = batch['labels'].to(device)
        #import pdb; pdb.set_trace()
        
        outputs = fbmodel(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels)
        
        
        
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        #pdb.set_trace()
