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
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from tools import *
from models import *
from dataset import ddDataset
from evaluation import evaluation

import argparse
import datetime

def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        type=str,
        #required=False,
        default='daily_dialog'
    )
    parser.add_argument(
        "--max_len", 
        type=int,
        #required=False,
        default=128,
        help="max len of sentence"
    )
    parser.add_argument(
        "--model_dir", 
        type=str,
        #required=False,
        default='./old/model_checkpoints/',
        help="directory of save models"
    )
    parser.add_argument(
        "--load_model_name", 
        type=str,
        #required=False,
        default='lr=1e-5_model.epoch_3',
        help="model name to be loaded, added to model_dir"
    )
    parser.add_argument(
        "--eval_sample_negatives", 
        #type=bool,
        #required=False,
        action='store_true',
        help="if add negative samples in calculating accuracy"
    )
    parser.add_argument(
        "--do_tsne", 
        #type=bool,
        #required=False,
        action='store_true'
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int,
        #required=False,
        default=3,
        help="train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int,
        #required=False,
        default=3,
        help="eval batch size"
    )
    parser.add_argument(
        "--test_batch_size", 
        type=int,
        required=False,
        default=3,
        help="test batch size"
    )
    parser.add_argument(
        "--k_neg_pos", "-k", 
        type=int,
        #required=False,
        default=3,
        help="negatives vs. positives"
    )
    parser.add_argument(
        "--FB_function_size", 
        type=int,
        #required=False,
        default=64,
        help="the size of the forward backward function"
    )
    parser.add_argument(
        "--tsne_plot_dir", 
        type=str,
        default='./tsne_plots/',
        help="directory to save tsne plots"
    )
    parser.add_argument(
        "--csv_dir", 
        type=str,
        default='./csvs/',
        help="directory to save csv file for F/B functions"
    )
    parser.add_argument(
        "--trial", 
        #type=bool,
        #required=False,
        action='store_true'
    )
    parser.add_argument(
        "--load_state_dict", 
        #type=bool,
        #required=False,
        action='store_true'
    )
    parser.add_argument(
        "--calculate_accuracy", 
        #type=bool,
        #required=False,
        action='store_true'
    )
    parser.add_argument(
        "--model_type", 
        type=str,
        #required=False,
        default='binary'
    )
    return parser

#%%
def get_dataset_acc(args, dataset, dialogs_flat, k, model, batch_size, device, sample_negatives, calculate_accuracy):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.eval()
    loop = tqdm(loader, leave=True)
    n_processed = 0
    total_correct = 0
    total_loss = 0
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bag_of_sents_tok = tokenizer(dialogs_flat, return_tensors='pt', max_length=args.max_len, truncation=True, padding='max_length')

    
    data_rep = dict()
    data_rep['true_labels'] = torch.LongTensor().cpu()
    data_rep['pred_labels'] = torch.LongTensor().cpu()
    data_rep['prev_forward'] = torch.Tensor().cpu()
    data_rep['curr_forward'] = torch.Tensor().cpu()
    data_rep['curr_backward'] = torch.Tensor().cpu()
    data_rep['next_backward'] = torch.Tensor().cpu()
    data_rep['input_ids_prev'] = torch.Tensor().cpu()
    data_rep['input_ids'] = torch.Tensor().cpu()
    data_rep['input_ids_next'] = torch.Tensor().cpu()
    data_rep['act'] = torch.Tensor().cpu()
    data_rep['act_prev'] = torch.Tensor().cpu()
    data_rep['act_next'] = torch.Tensor().cpu()
    data_rep['emotion'] = torch.Tensor().cpu()
    data_rep['emotion_prev'] = torch.Tensor().cpu()
    data_rep['emotion_next'] = torch.Tensor().cpu()
    
    for batch in loop:

        ############### negative samples ##########
        if len(batch['input_ids_prev']) != batch_size:
            break
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
        if calculate_accuracy:
            pred_labs = get_pred_labs(outputs.logits).to(device)
        #import pdb; pdb.set_trace()
        #labels = batch['true_labels_for_cal_acc'].to(device)
        #pdb.set_trace()
            if sample_negatives:
                labels = batch['true_labels_for_cal_acc'].to(device)
                n_correct = get_num_correct(pred_labs, labels)
            else:
                n_correct = get_num_correct(pred_labs, labels.repeat(2))
            total_correct += n_correct
            data_rep['pred_labels'] = torch.cat((data_rep['pred_labels'], pred_labs.cpu()), 0).cpu().detach()

        ####### add to return data ######
        data_rep['true_labels'] = torch.cat((data_rep['true_labels'], labels.cpu()), 0).cpu().detach()
        
        # [prev_forward, curr_forward, curr_backward, next_backward]
        data_rep['prev_forward'] = torch.cat((data_rep['prev_forward'], outputs.hidden_states[0].cpu()),0).cpu().detach()
        data_rep['curr_forward'] = torch.cat((data_rep['curr_forward'], outputs.hidden_states[1].cpu()),0).cpu().detach()
        data_rep['curr_backward'] = torch.cat((data_rep['curr_backward'], outputs.hidden_states[2].cpu()),0).cpu().detach()
        data_rep['next_backward'] = torch.cat((data_rep['next_backward'], outputs.hidden_states[3].cpu()),0).cpu().detach()
        data_rep['input_ids_prev'] = torch.cat((data_rep['input_ids_prev'], batch['input_ids_prev'].cpu()),0).cpu().detach()
        data_rep['input_ids'] = torch.cat((data_rep['input_ids'], batch['input_ids'].cpu()),0).cpu().detach()
        data_rep['input_ids_next'] = torch.cat((data_rep['input_ids_next'], batch['input_ids_next'].cpu()),0).cpu().detach()
        data_rep['act'] = torch.cat((data_rep['act'], batch['act'].cpu()),0).cpu().detach()
        data_rep['act_prev'] = torch.cat((data_rep['act_prev'], batch['act_prev'].cpu()),0).cpu().detach()
        data_rep['act_next'] = torch.cat((data_rep['act_next'], batch['act_next'].cpu()),0).cpu().detach()
        data_rep['emotion'] = torch.cat((data_rep['emotion'], batch['emotion'].cpu()),0).cpu().detach()
        data_rep['emotion_prev'] = torch.cat((data_rep['emotion_prev'], batch['emotion_prev'].cpu()),0).cpu().detach()
        data_rep['emotion_next'] = torch.cat((data_rep['emotion_next'], batch['emotion_next'].cpu()),0).cpu().detach()
        #################################
    print('eval loss: ', total_loss/n_processed)
    if calculate_accuracy:
        acc = total_correct/ n_processed
    else:
        acc = -100
    #pdb.set_trace()
    return acc, data_rep


def get_pred_labs (logits):
    pred = [1 if el[0] < el[1] else 0 for el in logits]
    return torch.LongTensor(pred)

def get_num_correct (pred_labs, true_labs):

    return (pred_labs == true_labs).float().sum()


def main():
    #%%
    now = datetime.datetime.now()
    curr_date = now.strftime("%Y-%m-%d")
    parser = get_parser()
    args = parser.parse_args()
    if args.do_tsne:
        print('do tsne')
    if args.eval_sample_negatives:
        print('eval negative sample')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    dataset = load_dataset(args.dataset)

    #dialogs = dataset['train']['dialog']
    #dialogs_eval = dataset['validation']['dialog']
    #dialogs_test = dataset['test']['dialog']

    dialogs = dataset['train']
    dialogs_eval = dataset['validation']
    dialogs_test = dataset['test']
    
    
    #if args.trial:
    #    dialogs = dialogs[:20]
    #    dialogs_eval = dialogs_eval[:20]
    #    dialogs_test = dialogs_test[:20]
    
    dialogs_flat = [utt for dialog in dialogs['dialog'] for utt in dialog]


    #[curr_sents, prev_sents, next_sents], acts, emotions = constructPositives_dataset(dialogs)
    #[curr_sents_eval, prev_sents_eval, next_sents_eval], acts_eval, emotions_eval = constructPositives_dataset(dialogs_eval)
    #[curr_sents_test, prev_sents_test, next_sents_test], acts_test, emotions_test = constructPositives_dataset(dialogs_test)
    sents, acts, emotions = constructPositives_dataset(dialogs)
    sents_eval, acts_eval, emotions_eval = constructPositives_dataset(dialogs_eval)
    sents_test, acts_test, emotions_test = constructPositives_dataset(dialogs_test)

    #pdb.set_trace()
    #ddtrain = constructInputs(prev_sents, curr_sents, next_sents, 'dailydialog')
    #ddeval = constructInputs(curr_sents_eval, prev_sents_eval, next_sents_eval, 'dailydialog')
    ddtest = constructInputs_with_act_emotion(sents_test, acts_test, emotions_test, 'dailydialog')
    print('added act and emotion')
    model_path = args.model_dir

    #model = AutoModel.from_pretrained('bert-base-uncased')
    # originally used BertForNextSentencePrediction
    #bertmodel = BertForForwardBackwardPrediction(model.config)
    if args.load_state_dict:
        model = AutoModel.from_pretrained('bert-base-uncased')
        model.config.__dict__['FB_function_size']=args.FB_function_size
        fbmodel = BertForForwardBackward_cos_flex(model.config)
        fbmodel.load_state_dict(torch.load(model_path+ args.load_model_name))
    else:
        fbmodel = torch.load( model_path+ args.load_model_name)
    print (fbmodel.config)
    #torch.save(fbmodel.state_dict(), model_path+ 'statedict_lr=1e-5_model.epoch_3')
    # lr=1e-5_model.epoch_3
    # 0.9793
    # lr=5e-6_model.epoch_3
    # 0.9679
    #pdb.set_trace()
    

    #loader = torch.utils.data.DataLoader(ddtrain, batch_size=args.train_batch_size, shuffle=True)
    #loader_eval = torch.utils.data.DataLoader(ddeval, batch_size=args.eval_batch_size, shuffle=True)
    loader_test = torch.utils.data.DataLoader(ddtest, batch_size=args.test_batch_size, shuffle=True)

    #acc_test, outputs = get_dataset_acc(ddtest, bertmodel, args.test_batch_size, device, True)
    #print('bert test acc: ',acc_test)

    #acc_eval, outputs = get_dataset_acc(args, ddeval, dialogs_flat, args.k_neg_pos, fbmodel, args.eval_batch_size, device, args.eval_sample_negatives)
    #print('eval acc: ',acc_eval)

    acc_test, outputs = get_dataset_acc(args, ddtest, dialogs_flat, args.k_neg_pos, fbmodel, args.eval_batch_size, device, args.eval_sample_negatives, args.calculate_accuracy)
    print('fbmodel test acc: ',acc_test)
    
    if args.do_tsne:
        time_start = time.time()
        #pdb.set_trace()
        df=pd.DataFrame(torch.cat((outputs['curr_backward'], outputs['prev_forward']), 0).cpu().detach().numpy())
        func = [0]*len(outputs['curr_backward']) + [1]*len(outputs['curr_backward'])
        df['function'] = pd.Series(np.array(func), index=df.index)
        #pdb.set_trace()
        df['act'] = pd.Series(outputs['act'].repeat(2).cpu().detach().numpy().astype(str))
        df['act_prev'] = pd.Series(outputs['act_prev'].repeat(2).cpu().detach().numpy().astype(str))
        df['emotion'] = pd.Series(outputs['emotion'].repeat(2).cpu().detach().numpy().astype(str))
        df['emotion_prev'] = pd.Series(outputs['emotion_prev'].repeat(2).cpu().detach().numpy().astype(str))
        df.to_csv(str(args.csv_dir)+curr_date+'FBsize='+str(args.FB_function_size)+'k='+str(args.k_neg_pos)+'.csv',
                  index = True)
        n_component = 2
        #pdb.set_trace()
        ppl = [ 50, 40, 30]
 
        #ppl = [50]
        num_iter = [ 1000, 300,500]
        #num_iter = [ 1000,500]
        #pdb.set_trace()
        for p in ppl:
            for it in num_iter:
                #from sklearn.manifold import TSNE
                tsne = TSNE(n_components=n_component, verbose=1, perplexity=p, n_iter=it)
                #tsne_results = tsne.fit_transform(df)
                tsne_results = tsne.fit_transform(df[df.columns[0:args.FB_function_size]])
                #print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
                #pdb.set_trace()
                if n_component == 2:
                    df['tsne-2d-one'] = tsne_results[:,0]
                    df['tsne-2d-two'] = tsne_results[:,1]
                    import matplotlib.pyplot as plt
                    plt.figure(0,figsize=(16,10))
                    sns.scatterplot(
                        x="tsne-2d-one", y="tsne-2d-two",
                        style="function",
                        hue = "act",
                            # palette=sns.color_palette("hls", 10),
                        data=df,
                        legend="full",
                        alpha=0.3
                    )
                    plt.show()
                    plt.savefig(str(args.tsne_plot_dir)+curr_date+ "curr_act" +"FBsize="+str(args.FB_function_size)+ "n_comp="+str(n_component)+"_ppl"+str(p)+"step"+str(it)+".png")
                    plt.clf()
                    plt.cla()
                    plt.close()
                    import matplotlib.pyplot as plt
                    plt.figure(1,figsize=(16,10))
                    sns.scatterplot(
                        x="tsne-2d-one", y="tsne-2d-two",
                        style="function",
                        hue = "act_prev",
                            # palette=sns.color_palette("hls", 10),
                        data=df,
                        legend="full",
                        alpha=0.3
                    )
                    plt.show()
                    plt.savefig(str(args.tsne_plot_dir)+curr_date+ "prev_act" +"FBsize="+str(args.FB_function_size)+ "n_comp="+str(n_component)+"_ppl"+str(p)+"step"+str(it)+".png")
                    plt.clf()
                    plt.cla()
                    plt.close()
                    import matplotlib.pyplot as plt
                    plt.figure(2,figsize=(16,10))
                    sns.scatterplot(
                        x="tsne-2d-one", y="tsne-2d-two",
                        style="function",
                        hue = "emotion",
                            # palette=sns.color_palette("hls", 10),
                        data=df,
                        legend="full",
                        alpha=0.3
                    )
                    plt.show()
                    plt.savefig(str(args.tsne_plot_dir)+curr_date+ "curr_emotion" +"FBsize="+str(args.FB_function_size)+ "n_comp="+str(n_component)+"_ppl"+str(p)+"step"+str(it)+".png")
                    plt.clf()
                    plt.cla()
                    plt.close()
                    import matplotlib.pyplot as plt
                    plt.figure(3,figsize=(16,10))
                    sns.scatterplot(
                        x="tsne-2d-one", y="tsne-2d-two",
                        style="function",
                        hue = "emotion_prev",
                            # palette=sns.color_palette("hls", 10),
                        data=df,
                        legend="full",
                        alpha=0.3
                    )
                    plt.show()
                    plt.savefig(str(args.tsne_plot_dir)+curr_date+ "prev_emotion" +"FBsize="+str(args.FB_function_size)+ "n_comp="+str(n_component)+"_ppl"+str(p)+"step"+str(it)+".png")
                    plt.clf()
                    plt.cla()
                    plt.close()
                elif n_component ==3:
                    df['tsne-3d-one'] = tsne_results[:,0]
                    df['tsne-3d-two'] = tsne_results[:,1]
                    df['tsne-3d-three'] = tsne_results[:,2]
                    plt.figure(figsize=(16,10))
                    sns.scatterplot(
                        x="tsne-3d-one", y="tsne-3d-two", z="tsne-3d-three",
                        hue="function",
                            # palette=sns.color_palette("hls", 10),
                        data=df,
                        legend="full",
                        alpha=0.3
                    )
                print ('Plotted perplexity %d step %d'%(p, it))
                print('Time plotting this plot: {} seconds'.format(time.time() - time_start))


    #pdb.set_trace()
    print('model: ', args.load_model_name)


#acc_train = get_dataset_acc(ddtrain, fbmodel, batch_size, device)
#print('train acc: ',acc_train)

if __name__ == "__main__":
    main()