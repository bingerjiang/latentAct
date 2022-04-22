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
from pytorch_metric_learning.losses import NTXentLoss

from tools import *
from models import *
from dataset import *
from test import *
from NCE import *
from cpc_train import *

@torch.no_grad()
def evaluation(model, loader_eval, device, epoch,  k, sample_negatives):
    
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
        loop.set_description(f"Epoch {epoch} Losss {total_loss/n_processed}")
        # *2 because the inputs is re-organized into 2* pairs
        # (prev, curr); (curr, next)
    # print('eval loss: ', total_loss/n_processed)
    # print(
    #     '| end of epoch {:3d} | time: {:5.2f}s |'
    #     'valid loss {:8.2f}'.format(
    #         epoch,
    #         (time.time() - epoch_start_time),
    #         total_loss/n_processed)
    # )
    
    return total_loss/n_processed

def eval_cpc (CPCmodel, encoding_model, loader_eval, device, epoch, args, logger ):
    CPCmodel.eval()
    encoding_model.eval()
    loop = tqdm(loader_eval, leave=True)
    total_loss = 0
    n_processed = 0
    n_correct = 0
    n_cosine_processed = 0
    epoch_start_time = time.time()
    for batch in loop:
        #optim.zero_grad()            
        with torch.no_grad():
            sent_embeddings = get_sentence_embeddings(encoding_model,batch, device)
        
            CPCmodel.to(device)
            if args.CPC_AR_model_type == 'transformer':
                sent_embeddings = sent_embeddings.view(-1, args.n_turns, 768)
                cFeatures = CPCmodel(sent_embeddings)
            else:
                cFeatures = CPCmodel(sent_embeddings)
            ## cFeatures.shape = [11, 11, 768]
            cFeatures_prediction = cFeatures[:, :-args.predict_k, :]
            ## predictions to be matched to input encodings 
            
            gar = sent_embeddings.view(-1, 11, 768)
            gar_target = gar[:, args.predict_k:, :]

            cFeatures_prediction = torch.flatten(cFeatures_prediction, start_dim=0, end_dim=1)
            gar_target = torch.flatten(gar_target, start_dim=0, end_dim=1)            
            #pdb.set_trace()
            assert(cFeatures_prediction.shape == gar_target.shape)
            
            embeddings = torch.cat((cFeatures_prediction, gar_target), dim=0)
            
            label_length = gar_target.shape[0]
            labels = torch.arange(label_length)
            labels = labels.repeat(2)
            
            loss_func = NTXentLoss()
            
            #testloss = True
            #if testloss:
            #    loss_func = NCEloss()
                
            nce_weights = torch.Tensor()
            
            loss = loss_func(embeddings, labels)
            
            total_loss+= loss.item()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            n_processed += len(batch['input_ids'])
            print('eval loss: ', total_loss/n_processed)
            print(
            '| end of epoch {:3d} | time: {:5.2f}s |'
            'valid loss {:8.2f}'.format(
                epoch,
                (time.time() - epoch_start_time),
                total_loss/n_processed)
            )
            pred_list, num_correct = get_top2_cosine_distance(embeddings)
            n_correct += num_correct
            n_cosine_processed += len(pred_list)
    
    epoch_time = time.time()-epoch_start_time
    print('eval time this epoch: {}'.format(epoch_time))
    logger.info('eval time this epoch: {}'.format(epoch_time))
    print('eval accuracy from cosine similarity: {:.4f}'.format(float(n_correct/n_cosine_processed)))
    logger.info('eval accuracy from cosine similarity: {:.4f}'.format(float(n_correct/n_cosine_processed)))
 
    return total_loss/n_processed


def eval_nsp(CPCmodel, encoding_model, loader_eval, device, epoch, args, logger, calculate_test_acc):
    epoch_start_time = time.time()
    loop = tqdm(loader_eval, leave=True)
    total_loss = 0
    n_processed = 0
    total_correct = 0
    n_correct = 0
    n_cosine_processed = 0
    n_processed_acc=0
    
    CPCmodel.eval()
    encoding_model.eval()
    loss_func = NCE_loss()
    loss_func.eval()
    for batch in loop:
            
        #sample negatives
        #neg_labs = [1]*k*2
        #pdb.set_trace()
        with torch.no_grad():
            sent_embeddings = get_sentence_embeddings(encoding_model, batch, device)
            
            CPCmodel.to(device)
           
            if args.CPC_AR_model_type == 'transformer':
                sent_embeddings = sent_embeddings.view(-1, args.n_turns, 768)
                cFeatures = CPCmodel(sent_embeddings)
            else:
                cFeatures = CPCmodel(sent_embeddings)

            cFeatures_prediction = cFeatures[:, :-args.predict_k, :]
            ## predictions to be matched to input encodings 
            
            gar = sent_embeddings.view(-1, 11, 768)
            gar_target = gar[:, args.predict_k:, :]
            
            actual_batch_size = gar.shape[0]

            cFeatures_prediction = torch.flatten(cFeatures_prediction, start_dim=0, end_dim=1)
            gar_target = torch.flatten(gar_target, start_dim=0, end_dim=1)            

            assert(cFeatures_prediction.shape == gar_target.shape)
            
            embeddings = torch.cat((cFeatures_prediction, gar_target), dim=0)
            #pdb.set_trace()
            
            label_length = gar_target.shape[0]
            labels = torch.arange(label_length)
            #labels = labels.repeat(2)
            #pdb.set_trace()
            
            loss_func.to(device)
            
            #loss_func = NTXentLoss()
            
            loss, logits, labels = loss_func(cFeatures_prediction, gar_target, actual_batch_size, args.n_turns,device)
            if calculate_test_acc:
                pred_labs = get_pred_labs(logits).to(device)
                n_correct = get_num_correct(pred_labs, labels)
                total_correct += n_correct
                

            total_loss+= loss.item()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            n_processed += len(batch['input_ids'])*10
            n_processed_acc += len(pred_labs)*10
                        ## see how many are correctly predicted to have the largest cos similarity
            pred_list, num_correct = get_top2_cosine_distance(embeddings)
            n_correct += num_correct
            n_cosine_processed += len(pred_list)
            pdb.set_trace()

    epoch_time = time.time()-epoch_start_time
    print('eval time this epoch: {}'.format(epoch_time))
    logger.info('eval time this epoch: {}'.format(epoch_time))
    print('eval accuracy from cosine similarity: {:.4f}'.format(float(n_correct/n_cosine_processed)))
    logger.info('eval accuracy from cosine similarity: {:.4f}'.format(float(n_correct/n_cosine_processed)))
    
    acc = -1
    if calculate_test_acc:
        acc = total_correct/ n_processed_acc
    return total_loss/n_processed, acc









