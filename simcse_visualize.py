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
from sklearn.cluster import KMeans
from vector_quantize_pytorch import VectorQuantize

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from tools import *
from models import *
from dataset import *
from evaluation import evaluation

import argparse
import datetime
import random

def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        type=str,
        #required=False,
        default='daily_dialog',
        help='choose among: daily_dialog, meta_woz'
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
        "--load_state_dict_new", 
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
        default='binary',
        help='choose among: binary, cos, cos_tlayer'
    )
    parser.add_argument(
        "--model", 
        type=str,
        #required=False,
        default='fb',
        help='choose among: fb, simcse'
    )
    parser.add_argument(
        "--pretrained", 
        action='store_true'
    )
    parser.add_argument(
        "--pretrained_model_name", 
        type=str,
        #required=False,
        default='princeton-nlp/unsup-simcse-bert-base-uncased'
    )
    parser.add_argument(
        "--make_plot", 
        #type=bool,
        #required=False,
        action='store_true'
    )
    parser.add_argument(
        "--save_csv", 
        #type=bool,
        #required=False,
        action='store_true'
    )
    parser.add_argument(
        "--plot_kmeans_tsne", 
        #type=bool,
        #required=False,
        action='store_true'
    )
    parser.add_argument(
        "--pre_z_tanh", 
        action='store_true'
    )
    parser.add_argument(
        "--vector_quantize", 
        action='store_true'
    )
    parser.add_argument(
        "--test_k",  
        type=int,
        #required=False,
        default=1,
        help="negatives vs. positives for eval/test"
    )
    return parser

#%%
def get_simcse_embed(args, dataset,  tokenizer, k, model, model_name, batch_size, device, sample_negatives, calculate_accuracy):
    #dialogs_test_flat = [utt for dialog in dialogs_test['dialog'] for utt in dialog]
    #inputs = tokenizer(dialogs_test_flat, padding=True, truncation=True, return_tensors="pt")
    #outputs = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.eval()
    loop = tqdm(loader, leave=True)
    n_processed = 0
    total_correct = 0
    total_loss = 0
    all_outputs = torch.Tensor()
    for batch in loop:
        #pdb.set_trace()
        #batch.to(device)
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids,attention_mask=attention_mask,
                            token_type_ids=token_type_ids, 
                            output_hidden_states=True, return_dict=True).pooler_output
            
            n_processed += len(batch['input_ids'])*2 
            all_outputs = torch.cat((all_outputs, outputs.cpu()),0).cpu().detach()



    print('eval loss: ', total_loss/n_processed)

    return all_outputs
def get_cpc_embed(args, dataset,  model, model_name, batch_size, device):
    #dialogs_test_flat = [utt for dialog in dialogs_test['dialog'] for utt in dialog]
    #inputs = tokenizer(dialogs_test_flat, padding=True, truncation=True, return_tensors="pt")
    #outputs = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    loop = tqdm(loader, leave=True)
    n_processed = 0
    total_correct = 0
    total_loss = 0
    all_outputs = torch.Tensor()
    for batch in loop:
        #pdb.set_trace()
        #batch.to(device)
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sent_embeddings = sbert(encoding_model,batch, device)
            
            n_processed += len(batch['input_ids'])*2 
            all_outputs = torch.cat((all_outputs, outputs.cpu()),0).cpu().detach()



    print('eval loss: ', total_loss/n_processed)

    return all_outputs

def get_pred_labs (logits):
    pred = [1 if el[0] < el[1] else 0 for el in logits]
    return torch.LongTensor(pred)

def get_num_correct (pred_labs, true_labs):

    return (pred_labs == true_labs).float().sum()


def main():
    #%%
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    now = datetime.datetime.now()
    curr_date = now.strftime("%Y-%m-%d")
    parser = get_parser()
    args = parser.parse_args()
    if args.do_tsne:
        print('do tsne')
    if args.eval_sample_negatives:
        print('eval negative sample')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if args.model == 'simcse':
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        fbmodel = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    
    ## load datasets
    dataset = load_dataset(args.dataset)
    dialogs = dataset['train']
    dialogs_test = dataset['test']
    

    if args.dataset == 'daily_dialog': 
        dialogs_test_flat = [utt for dialog in dialogs_test['dialog'] for utt in dialog]
        tokenized_inputs = tokenizer(dialogs_test_flat, padding=True, truncation=True, return_tensors="pt")
        ddtest = initializeDataset(tokenized_inputs)
    elif args.dataset == 'meta_woz': 
        dialogs_flat = [utt for dialog in dialogs['turns'] for utt in dialog]
        sents_test, acts_test, emotions_test = constructPositives(dialogs_test)
        ddtest = constructInputs(sents_test, acts_test, emotions_test, args.dataset)

    
    print (fbmodel.config)
    #pdb.set_trace()
    loader_test = torch.utils.data.DataLoader(ddtest, batch_size=args.test_batch_size, shuffle=True)

    if args.model == 'simcse':
        outputs = get_simcse_embed(args, ddtest, tokenizer, args.test_k, fbmodel, args.model, args.eval_batch_size, device, args.eval_sample_negatives, args.calculate_accuracy)
       # with torch.no_grad():
            # simCSE requires flat list
        #    dialogs_test_flat = [utt for dialog in dialogs_test['dialog'] for utt in dialog]
        #    inputs = tokenizer(dialogs_test_flat, padding=True, truncation=True, return_tensors="pt")
        #    outputs = fbmodel(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    elif args.model == 'fb':
        acc_test, outputs = get_dataset_acc(args, ddtest, dialogs_flat, tokenizer, args.test_k, fbmodel, args.model, args.eval_batch_size, device, args.eval_sample_negatives, args.calculate_accuracy)
        print('fbmodel test acc: ',acc_test)

    if args.vector_quantize:
        print ('vector quantization')
        vq = VectorQuantize(dim = outputs.shape[-1],codebook_size = 126,decay = 0.8)        
        
        quantized, indices, commit_loss = vq(outputs) 
            #pdb.set_trace()
        print(set(list(indices.cpu().detach().numpy())))
        print('length of indices vq labels:', len(set(list(indices.cpu().detach().numpy()))))
        longest, longest_n = 0, 0
        for n in range(5,512):
            vq = VectorQuantize(dim = outputs.shape[-1],codebook_size = n,decay = 0.8)        
            
            quantized, indices, commit_loss = vq(outputs) 
            #pdb.set_trace()
            print(set(list(indices.cpu().detach().numpy())))
            print('length of indices vq labels:', len(set(list(indices.cpu().detach().numpy()))))
            if len(set(list(indices.cpu().detach().numpy()))) > longest:
                longest =len(set(list(indices.cpu().detach().numpy())))
                longest_n = n
        print('longest: ', longest)
        print('longest_n: ', longest_n)
    if args.make_plot:
        # TODO: missing mapping to sentences
        time_start = time.time()
        df = pd.DataFrame(outputs.cpu().detach().numpy())
        if args.vector_quantize:
            df['vq_lab'] = pd.Series(indices.cpu().detach().numpy().astype(str))
         
        X = df[df.columns[1:outputs.shape[-1]+1]]
        #pdb.set_trace()
        kmeans = KMeans(n_clusters=14, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )
        kmeans.fit(X)
        y_kmeans = kmeans.fit_predict(X)
        print(set(y_kmeans))
        df['kmeans_lab'] = y_kmeans.astype(str)
        
        if args.save_csv:
            df.to_csv(str(args.csv_dir)+curr_date+str(args.model_type)+\
                str(args.dataset)+'.csv',
                  index = True)
        
        #pdb.set_trace()
    n_component = 2
    ppl = [ 50, 40, 30]
 
        #ppl = [50]
    num_iter = [ 1000, 300,500]
    
    if args.plot_kmeans_tsne:
        for p in ppl:
            for it in num_iter:
                #from sklearn.manifold import TSNE
                tsne = TSNE(n_components=n_component, verbose=1, perplexity=p, n_iter=it)
                #tsne_results = tsne.fit_transform(df)
                tsne_results = tsne.fit_transform(df[df.columns[1:outputs.shape[-1]+1]])
                #print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
                #pdb.set_trace()
                if n_component == 2:
                    df['tsne-2d-one'] = tsne_results[:,0]
                    df['tsne-2d-two'] = tsne_results[:,1]
                    import matplotlib.pyplot as plt
                    
                    
                    plt.figure(figsize=(16,10))
                    sns.scatterplot(
                        x="tsne-2d-one", y="tsne-2d-two",
                        #style="function",
                        hue = "kmeans_lab",
                            # palette=sns.color_palette("hls", 10),
                        data=df,
                        legend="full",
                        alpha=0.3
                    )
                    plt.show()
                    plt.savefig(str(args.tsne_plot_dir)+curr_date+'kmeans_'+str(args.model_type)+"_FBsize="+\
                        str(args.FB_function_size)+ str(args.dataset)+ \
                        "_n_comp="+str(n_component)+"_ppl"+str(p)+"step"+str(it)+".png")
                    plt.clf()
                    plt.cla()
                    plt.close()
                    plt.figure(figsize=(16,10))
                    sns.scatterplot(
                        x="tsne-2d-one", y="tsne-2d-two",
                        #style="function",
                        hue = "vq_lab",
                            # palette=sns.color_palette("hls", 10),
                        data=df,
                        legend="full",
                        alpha=0.3
                    )
                    plt.show()
                    plt.savefig(str(args.tsne_plot_dir)+curr_date+'vq_'+str(args.model_type)+"_FBsize="+\
                        str(args.FB_function_size)+ str(args.dataset)+ \
                        "_n_comp="+str(n_component)+"_ppl"+str(p)+"step"+str(it)+".png")
    
    if args.do_tsne:
        
        #num_iter = [ 1000,500]
        #pdb.set_trace()
    
        for p in ppl:
            for it in num_iter:
                #from sklearn.manifold import TSNE
                tsne = TSNE(n_components=n_component, verbose=1, perplexity=p, n_iter=it)
                #tsne_results = tsne.fit_transform(df)
                tsne_results = tsne.fit_transform(df[df.columns[1:args.FB_function_size+1]])
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
                    plt.savefig(str(args.tsne_plot_dir)+curr_date+str(args.dataset) +'_'+ "curr_act_" +str(args.model_type)+"_FBsize="+\
                        str(args.FB_function_size)+ str(args.dataset)+ \
                            + "_n_comp="+str(n_component)+"_ppl"+str(p)+"step"+str(it)+".png")
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
                    plt.savefig(str(args.tsne_plot_dir)+curr_date+ str(args.dataset) +'_'+"prev_act_" +str(args.model_type)+"_FBsize="+\
                        str(args.FB_function_size)+ str(args.dataset)+ \
                            + "_n_comp="+str(n_component)+"_ppl"+str(p)+"step"+str(it)+".png")
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
                    plt.savefig(str(args.tsne_plot_dir)+curr_date+ "curr_emotion_" +str(args.model_type)+"_FBsize="+\
                        str(args.FB_function_size)+ str(args.dataset)+ \
                            + "_n_comp="+str(n_component)+"_ppl"+str(p)+"step"+str(it)+".png")
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
                    plt.savefig(str(args.tsne_plot_dir)+curr_date+ str(args.dataset) +'_'+"prev_emotion_" +str(args.model_type)+"_FBsize="+\
                        str(args.FB_function_size)+ str(args.dataset)+ \
                             "_n_comp="+str(n_component)+"_ppl"+str(p)+"step"+str(it)+".png")
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