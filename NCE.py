import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
from models import *

from test import *
#%%
#logger = setup_logger('{}'.format('model.pt'))
import datetime
import argparse
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import  BertForPreTraining, BertForSequenceClassification, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import NextSentencePredictorOutput

import pdb


class NCE_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = nn.Linear(768*2, 2)

    def forward(self, cFeatures, z, batch_size, n_turns, device):
        
        ## pos_ .shape: (batch_size, dim)
        # pos_1: cfeature
        # pos_2: z (negs to be sampled from)
    
        ####### use all ten other utterences in the turn as negatives #####
        ## assuming 11 turns
        #print('This assumes 11 turns; needs change if not')
        #print('using other 10 utts as negs in a conversation')
        turn_idx = np.arange(0,n_turns-1) # 11 turns, 10 to be predicted
        base_negs = np.array([np.delete(turn_idx,el) for el in turn_idx])
        ###############################
        neg_idx_all = base_negs
        
        #batch_size = 6, but may be smaller!!! not enough data --> careful
        #n_turns = 11
        i = 1
        while i <batch_size:
            l_new = base_negs+i*(n_turns-1)
            neg_idx_all = np.concatenate([neg_idx_all, l_new], axis=None)
            i+=1
        #print('----Done getting neg indices---')
        ###### done negatives ######
        
        ## the below 2 lines causes bug, index select is problematics
        #neg_idx_all = torch.from_numpy(neg_idx_all).to(device)
        #pdb.set_trace()
       # negs_all = torch.index_select(z, 0, neg_idx_all)
        #############
        #pdb.set_trace()
        base_negs = torch.from_numpy(base_negs).view(-1).to(device)
        negs_all = torch.index_select(z, 0, base_negs)
        
        i = 1
        while i <batch_size:
            
            neg_idx_new = base_negs+i*(n_turns-1)
            #print(i)
            #pdb.set_trace()
            new_negs = torch.index_select(z, 0, neg_idx_new)
            negs_all = torch.cat((negs_all, new_negs),0)
            
            i+=1
        
        # negs_all = z.repeat(n_turns-2,1) ## just for test run, debug purpose
        #negs_all = (s.detach() for s in negs_all)
        
        pos_embeddings = torch.cat((cFeatures, z), dim=1)
        #pdb.set_trace()
        cFeatures_negs = cFeatures.repeat(n_turns-2,1)
        #cFeatures_negs = (s.detach() for s in cFeatures_negs)
        #pdb.set_trace()
        #with torch.no_grad():
        neg_embeddings = torch.cat((cFeatures_negs, negs_all),dim=1)
        #neg_embeddings = pos_embeddings.repeat(n_turns-1)
        #neg_embeddings.requires_grad=False
        
        #neg_embeddings =  neg_embeddings.detach()
        
        # pos_labels = torch.zeros(z.shape[0], dtype=torch.float32, requires_grad=True)
        # neg_labels = torch.zeros(z.shape[0]*(n_turns-1),dtype=torch.float32, requires_grad=False)+1
        
        pos_labels = torch.zeros(z.shape[0])
        neg_labels = torch.zeros(z.shape[0]*(n_turns-2))+1
        pos_labels = pos_labels.type(torch.LongTensor)
        neg_labels = neg_labels.type(torch.LongTensor)
        
        #labels = torch.LongTensor([0]*len(curr_sents)).T
        #pdb.set_trace()
        pos_score = self.cls(pos_embeddings)
        neg_score = self.cls(neg_embeddings)
        
        loss_func = CrossEntropyLoss()
        #pdb.set_trace()
        pos_labels = pos_labels.to(device)
        neg_labels = neg_labels.to(device)
        
        pos_loss = loss_func(pos_score.view(-1,2), pos_labels.view(-1))
        neg_loss = loss_func(neg_score.view(-1,2), neg_labels.view(-1))

        all_scores = torch.cat((pos_score, neg_score), dim=0)
        all_labels = torch.cat((pos_labels, neg_labels), dim=0)
        return pos_loss + neg_loss, all_scores, all_labels
        