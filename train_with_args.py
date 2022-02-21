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
from models import *
from dataset import ddDataset
from evaluation import evaluation
from test import *
#%%
#logger = setup_logger('{}'.format('model.pt'))
import datetime
import argparse
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
        "--model_type", 
        type=str,
        #required=False,
        default='binary'
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
        help="directory of saved models"
    )
    parser.add_argument(
        "--continue_train", 
        #type=bool,
        #required=False,
        action='store_true',
        help="if continue trainig from an existing model"
    )
    parser.add_argument(
        "--load_model_name", 
        type=str,
        #required=False,
        default='lr=1e-5_model.epoch_3',
        help="model name to be loaded, added to model_dir"
    )
    parser.add_argument(
        "--save_model_dir", 
        type=str,
        #required=False,
        default='./model_checkpoints/',
        help="model dir to save model"
    )
    parser.add_argument(
        "--save_model_name", 
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
        default=2,
        help="train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int,
        #required=False,
        default=8,
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
        default=10,
        help="negatives vs. positives"
    )
    parser.add_argument(
        "--test_k",  
        type=int,
        #required=False,
        default=1,
        help="negatives vs. positives for eval/test"
    )
    parser.add_argument(
        "--lr",  
        type=float,
        #required=False,
        default=1e-6,
        help="learning rate"
    )
    parser.add_argument(
        "--n_epochs", 
        type=int,
        #required=False,
        default=2,
        help="number of training epochs"
    )
    parser.add_argument(
        "--trial", 
        #type=bool,
        #required=False,
        action='store_true'
    )
    parser.add_argument(
        "--lr_decay",  
        type=float,
        #required=False,
        default=5,
        help="lr decay"
    )
    parser.add_argument(
        "--calculate_test_acc", 
        action='store_true'
    )
    parser.add_argument(
        "--FB_function_size", 
        type=int,
        #required=False,
        default=64,
        help="the size of the forward backward function"
    )
    parser.add_argument(
        "--flex", 
        action='store_true'
    )
    parser.add_argument(
        "--pre_z_tanh", 
        action='store_true'
    )
    #parser.add_argument(
    #    "--tlayer_size", 
    #    type=int,
    #    #required=False,
    #    default=240,
    #    help="tlayer size in loss"
    #)
    return parser


#%%
def main():
    parser = get_parser()
    args = parser.parse_args()
    
    dataset = load_dataset(args.dataset)
    dd_train = dataset['train']
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dialogs = dd_train['dialog']
    dialogs_eval = dataset['validation']['dialog']
    dialogs_test = dataset['test']['dialog']
    if args.trial:
        dialogs = dialogs[:20]
        dialogs_eval = dialogs_eval[:20]
    print('length of training data: ', len(dialogs))
    if args.eval_sample_negatives:
        print('eval sample negativs')
    dialogs_flat = [utt for dialog in dialogs for utt in dialog]

    bag_of_sents_tok = tokenizer(dialogs_flat, return_tensors='pt', max_length=args.max_len, truncation=True, padding='max_length')

    curr_sents, prev_sents, next_sents = constructPositives(dialogs)
    curr_sents_eval, prev_sents_eval, next_sents_eval = constructPositives(dialogs_eval)
    curr_sents_test, prev_sents_test, next_sents_test = constructPositives(dialogs_test)


    ddtrain = constructInputs(prev_sents, curr_sents, next_sents, 'dailydialog')
    ddeval = constructInputs(curr_sents_eval, prev_sents_eval, next_sents_eval, 'dailydialog')
    ddtest = constructInputs(curr_sents_test, prev_sents_test, next_sents_test, 'dailydialog')
    #test_input_prev = tokenizer(all_prev_sents[:50], return_tensors='pt', max_length=256, truncation=True, padding='max_length')
    #test_input_next = tokenizer(all_next_sents[:50], return_tensors='pt', max_length=256, truncation=True, padding='max_length')
    #test_labs = torch.LongTensor(all_labs[:50]).T
    #%%

    #%%
    model = AutoModel.from_pretrained('bert-base-uncased')
    
    ## TODO add
    
    model.config.__dict__['pre_z_tanh']=args.pre_z_tanh
    
    # originally used BertForNextSentencePrediction
    if args.model_type == 'binary':
        fbmodel = BertForForwardBackwardPrediction(model.config)
        if args.continue_train:
            fbmodel = torch.load(args.model_dir+ args.load_model_name)
        if args.flex:
            model.config.__dict__['FB_function_size']=args.FB_function_size
            #model.config.__dict__['tlayer_size']=args.tlayer_size
            fbmodel = BertForForwardBackward_binary_flex(model.config)
                
    elif args.model_type == 'cos':
        fbmodel = BertForForwardBackwardPrediction_cos(model.config)
        if args.flex:
            model.config.__dict__['FB_function_size']=args.FB_function_size
            #model.config.__dict__['tlayer_size']=args.tlayer_size
            fbmodel = BertForForwardBackward_cos_flex(model.config)
            print(model.config)
        #if args.trial:
        #    fbmodel = BertForForwardBackwardPrediction_cos(model.config)
    elif args.model_type == 'cos_tlayer':
        fbmodel = BertForForwardBackwardPrediction_cos_tlayer(model.config)
        if args.flex:
            model.config.__dict__['FB_function_size']=args.FB_function_size
            #model.config.__dict__['tlayer_size']=args.tlayer_size
            fbmodel = BertForForwardBackward_cos_tlayer_flex(model.config)
    
    print('fbmodel.config: ')
    print(fbmodel.config)
    batch_size_train= args.train_batch_size
    batch_size_eval = args.eval_batch_size
    batch_size_test = args.test_batch_size
    
    loader = torch.utils.data.DataLoader(ddtrain, batch_size=args.train_batch_size, shuffle=True)
    loader_eval = torch.utils.data.DataLoader(ddeval, batch_size=args.eval_batch_size, shuffle=True)
    loader_test = torch.utils.data.DataLoader(ddtest, batch_size=args.test_batch_size, shuffle=True)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fbmodel.to(device)
    fbmodel.train()
    lr = args.lr
    optim = AdamW(fbmodel.parameters(), lr=lr)

    model_path = args.model_dir
    lr_decay = args.lr_decay
    best_eval_loss = 100000

    n_plateau= 0
    epochs = args.n_epochs 
    k =args.k_neg_pos
    print('k: ', k)
    
    ###################
    ## print a set of hyperparams
    print('model and training info:')
    print('k_neg_pos= ', args.k_neg_pos)
    print('n_epochs= ', args.n_epochs)
    print('eval_batch_size= ', args.eval_batch_size) 
    print('train_batch_size= ', args.train_batch_size)
    print('model_type= ', args.model_type)
    print('lr= ', args.lr)
    print('save_model_dir= ', args.save_model_dir)
    
    
    #print('accuracy before training: ')
    #acc_test, outputs = get_dataset_acc(args, ddtest, dialogs_flat, args.test_k, fbmodel, args.test_batch_size, device, args.eval_sample_negatives)
    #print('test acc: ',acc_test)
    print('---- start training -----')
    for epoch in range(epochs):        
        loop = tqdm(loader, leave=True)
        total_loss = 0
        n_processed = 0
        for batch in loop:
            optim.zero_grad()            
            #sample negatives
            #neg_labs = [1]*k*2
            neg_labs = [1]
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

            # batch['labels']=torch.cat((batch['labels'],torch.LongTensor(neg_labs).T), 0)
            batch['labels']=torch.cat((batch['labels'],torch.LongTensor(neg_labs).T.repeat(k*batch_size_train)), 0)

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
        eval_loss = evaluation(fbmodel, loader_eval, device, epoch,args.test_k, args.eval_sample_negatives)
        print('args.calculate_test_acc: ', args.calculate_test_acc)
        if args.calculate_test_acc:
            acc_eval, outputs = get_dataset_acc(args, ddeval, dialogs_flat, args.test_k, fbmodel, args.eval_batch_size, device, args.eval_sample_negatives)
            print('eval acc: ',acc_eval)
        
        #torch.save(fbmodel, model_path + '.epoch_{}'.format(epoch))
        # Save the model if the validation loss is the best we've seen so far.
        if  eval_loss < best_eval_loss:
            if not args.trial:
                now = datetime.datetime.now()
                curr_date = now.strftime("%Y-%m-%d")
                    #torch.save(model.state_dict(), PATH)
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': fbmodel.state_dict(),
                            'optimizer_state_dict': optim.state_dict(),
                            'loss': eval_loss,
                            },  args.save_model_dir +curr_date + str(args.model_type)+\
                            'FB='+str(args.FB_function_size) +'_lr='+str(lr)+'_model''.epoch_{}'.format(epoch)+'.pt')
                #torch.save(fbmodel.state_dict(), args.save_model_dir +curr_date + str(args.model_type)+\
                #           'FB='+str(args.FB_function_size) +'_lr='+str(args.lr)+'_model''.epoch_{}'.format(epoch)+'.pt')
            best_eval_loss = eval_loss
            n_plateau = 0
            
        elif n_plateau >2:
            # Anneal the learning rate if no improvement has been seen in the
            # validation dataset.
            lr /= lr_decay
            print('lr decayed to: ', lr)
            n_plateau = 0
        else:
            n_plateau +=1
    if args.calculate_test_acc:
        acc_test, outputs = get_dataset_acc(args, ddtest, dialogs_flat, args.test_k, fbmodel, args.test_batch_size, device, args.eval_sample_negatives)
        print('test acc: ',acc_test)
# test
    
#%%
# k = 10

# nsp_a, nsp_b, nsp_labs = sample_next(dialogs_flat, k)
# psp_a, psp_b, psp_labs = sample_previous(dialogs_flat, k)


# # TODO: remove duplicate rows
# all_prev_sents = nsp_a + psp_b
# all_next_sents = nsp_b + psp_a
# all_labs = nsp_labs + psp_labs
if __name__ == "__main__":
    main()