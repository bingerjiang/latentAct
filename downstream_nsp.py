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
import random

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
from evaluation import *
from test import *
from NCE import *
#%%
#logger = setup_logger('{}'.format('model.pt'))
import datetime
import argparse

import logging
def get_parser():
    """Get argument parser."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_transformer_layer', type=int, default=1,
                       help='number of transformer layers in the transformer AR')

    parser.add_argument(
        "--CPC_AR_model_type", 
        type=str,
        default='transformer',
        help='choose between: transformer, GRU, LSTM, RNN'
    )
    ## cpc newly added part
    parser.add_argument(
        "--unfreeze_cpc", 
        action='store_true',
        help="if unfreeze cpc model"
    )
    parser.add_argument(
        "--load_from_cpc_pretrained", 
        action='store_true',
        help="if load from pretrained cpc for downstream"
    )
    parser.add_argument(
        "--use_cpc_nlp_optim", 
        action='store_true',
        help="if use the optim setting from CPC-NLP-Pytorch"
    )
    
    parser.add_argument(
        "--finetune_sent_encoding", 
        action='store_true',
        help="if finetune the encoding model"
    )
    parser.add_argument(
        "--sent_encoding_model", 
        type=str,
        default='sbert',
        help='choose between: sbert, simcse'
    )
    parser.add_argument('--n_turns', type=int, default=11,
                       help='batch size, actual batch size is batch_size*n_turns.')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='batch size, actual batch size is batch_size*n_turns.')
    parser.add_argument('--predict_k', type=int, default=1,
                       help='the k steps to be predicted in CPC model.')
    parser.add_argument('--nLevelsGRU', type=int, default=1,
                       help='Number of layers in the autoregressive network.')
    parser.add_argument('--dimEncoded', type=int, default=768,
                       help='Hidden dimension of the encoder network.')#dimEncoded
    parser.add_argument('--hiddenGar', type=int, default=768,
                       help='Hidden dimension of the auto-regressive network')#dimOutput
    ## end cpc newly added part
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
        "--load_model_dir", 
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
        default='../cpc_models/',
        help="model dir to save model"
    )
    parser.add_argument(
        "--save_model_name", 
        type=str,
        #required=False,
        default='unnamed',
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
        "--trial_save_model", 
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
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    logger_name = args.save_model_name
    if logger_name != 'unnamed':
        logger = setup_logs('./logs/', logger_name)
    else:
        logger = setup_logs('./logs/')
    logger.info('### args summary below ###\n {}'.format(args))
    #pdb.set_trace()
    
    max_len = args.max_len
    
        ## load datasets
    dataset = load_dataset(args.dataset)
    dialogs = dataset['train']
    dialogs_test = dataset['test']
   # pdb.set_trace()
    if args.dataset == 'meta_woz':
        ## remove domain == AGREEMENT_BOT because conversations are too weird
        agreebot_idx = len(dialogs.filter(lambda example: example['domain'].startswith('AGREE')))
        dialogs = dialogs[agreebot_idx:]
        
        
        split = int(0.9*len(dialogs['turns']))
        dialogs_train = dialogs['turns'][:split]
        dialogs_eval = dialogs['turns'][split:]
        dialogs_test = dialogs_test['turns']
        #pdb.set_trace()
        ## for simplicity, only using dialogs with 11 turns
        dialogs_train_truncate = [el for el in dialogs_train if len(el)>10]
        dialogs_train_11 = [el[:11] for el in dialogs_train_truncate]
        
        dialogs_eval_truncate = [el for el in dialogs_eval if len(el)>10]
        dialogs_eval_11 = [el[:11] for el in dialogs_eval_truncate]
        
        dialogs_test_truncate = [el for el in dialogs_test if len(el)>10]
        dialogs_test_11 = [el[:11] for el in dialogs_test_truncate]
        print('Warning by B2:')
        print('for simplicity, and most dialogs in metawoz is 11 turns')
        print('setting all dialogs to be 11 turns (cut otherwize), maxlen =128')
        
        ## make flat, then change shape later
        train_flat = [utt for dialog in dialogs_train_11 for utt in dialog]
        eval_flat = [utt for dialog in dialogs_eval_11 for utt in dialog]
        test_flat = [utt for dialog in dialogs_test_11 for utt in dialog]
        
        if args.trial:
            train_flat = train_flat[:121]
            eval_flat = eval_flat[:121]
            test_flat = test_flat[:121]
        
        ## initiate inputs in dataset format
        
        train_flat = initiateTokenizedInputs(train_flat, args.sent_encoding_model)
        eval_flat = initiateTokenizedInputs(eval_flat, args.sent_encoding_model)
        test_flat = initiateTokenizedInputs(test_flat, args.sent_encoding_model)
        # pdb.set_trace()
        
    
    
    # model = AutoModel.from_pretrained('bert-base-uncased')
    

    
    batch_size_train= args.train_batch_size
    batch_size_eval = args.eval_batch_size
    batch_size_test = args.test_batch_size
    
    loader = torch.utils.data.DataLoader(train_flat, batch_size= args.batch_size * args.n_turns, shuffle=False)
    loader_eval = torch.utils.data.DataLoader(eval_flat, batch_size= args.batch_size * args.n_turns, shuffle=False)
    loader_test = torch.utils.data.DataLoader(test_flat, batch_size= args.batch_size * args.n_turns, shuffle=False)
    
    if args.sent_encoding_model =='sbert':
        encoding_model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    elif args.sent_encoding_model == 'simcse':
        encoding_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    if args.CPC_AR_model_type == 'transformer':
        CPCmodel = buildTransformerAR(args.dimEncoded, args.n_transformer_layer, args.n_turns, False)
    else:
        CPCmodel = CPCAR(args.dimEncoded, args.hiddenGar, False, args.nLevelsGRU)

    lr = args.lr
    lr_decay = args.lr_decay
    
    model_params = CPCmodel.parameters()
    if args.finetune_sent_encoding:
        model_params = list(CPCmodel.parameters()) + list(CPCmodel.parameters())
    
    if args.use_cpc_nlp_optim:
        optim = torch.optim.Adam(model_params, 
            lr=2e-4, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True)
    else:
        optim = AdamW(model_params, lr=lr)


    loss_func = NCE_loss()
    
    best_eval_loss = 100000
    best_model_epoch = -1

    n_plateau= 0
    epochs = args.n_epochs 
    k =args.k_neg_pos
    # print('k: ', k)
    
    ###################
    ## print a set of hyperparams
    CPCmodel.to(device)
    encoding_model.to(device)
    
    if args.load_from_cpc_pretrained:
        PATH = args.load_model_dir +args.load_model_name+'-best_model.pt'
        checkpoint = torch.load(PATH)
        CPCmodel.load_state_dict(checkpoint['model_state_dict'])
        encoding_model.load_state_dict(checkpoint['encoder_model_state_dict'])
        #optim.load_state_dict(checkpoint['optimizer_state_dict'])
        #epoch = checkpoint['epoch']
        #best_eval_loss = checkpoint['loss']
        if 'loss_state_dict' in checkpoint.keys():
            loss_func.load_state_dict(checkpoint['loss_state_dict'])

    #print('accuracy before training: ')
    #acc_test, outputs = get_dataset_acc(args, ddtest, dialogs_flat, args.test_k, fbmodel, args.test_batch_size, device, args.eval_sample_negatives)
    #print('test acc: ',acc_test)
    print('---- start training -----')
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        loop = tqdm(loader, leave=True)
        total_loss = 0
        n_processed = 0
        #CPCmodel.train()
        loss_func.train()
        if args.finetune_sent_encoding:
            encoding_model.train()
        if args.unfreeze_cpc:
            CPCmodel.train()

        for batch in loop:
            optim.zero_grad()            
            #sample negatives
            #neg_labs = [1]*k*2
            #pdb.set_trace()
            
            neg_labs = [1]
            i = 0
            negatives = []
            ## sentence bert is not optimizing!!!
            if args.finetune_sent_encoding:
                sent_embeddings = get_sentence_embeddings(encoding_model, batch, device)
            else:
                with torch.no_grad():
                    sent_embeddings = get_sentence_embeddings(encoding_model, batch, device)
                    
            if args.CPC_AR_model_type == 'transformer':
                sent_embeddings = sent_embeddings.view(-1, args.n_turns, 768)

            if args.unfreeze_cpc:
                cFeatures = CPCmodel(sent_embeddings)
            else:
                with torch.no_grad():
                    cFeatures = CPCmodel(sent_embeddings)
                    
            
            
            #pdb.set_trace()
            ## cFeatures.shape = [batch_size, 11, 768]
            cFeatures_prediction = cFeatures[:, :-args.predict_k, :]
            ## predictions to be matched to input encodings 
            
            gar = sent_embeddings.view(-1, 11, 768)
            gar_target = gar[:, args.predict_k:, :]
            
            actual_batch_size = gar.shape[0]

            cFeatures_prediction = torch.flatten(cFeatures_prediction, start_dim=0, end_dim=1)
            gar_target = torch.flatten(gar_target, start_dim=0, end_dim=1)            

            assert(cFeatures_prediction.shape == gar_target.shape)
            
            #embeddings = torch.cat((cFeatures_prediction, gar_target), dim=0)
            #pdb.set_trace()
            
            label_length = gar_target.shape[0]
            labels = torch.arange(label_length)
            #labels = labels.repeat(2)
            #pdb.set_trace()
            
            loss_func.to(device)
 
            #loss_func = NTXentLoss()
           
            loss, _, _ = loss_func(cFeatures_prediction, gar_target, actual_batch_size, args.n_turns,device)
            
            total_loss+= loss.item()

            loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            n_processed += len(batch['input_ids']) 

        epoch_time = time.time() - epoch_start_time
        print('training loss: ', total_loss/n_processed)
        print('training time this epoch: {}'.format(epoch_time))
        logger.info(' ######## Train Epoch: {} ######## '.format(epoch))
        logger.info('Training loss: {:.4f}'.format(total_loss/n_processed))
        logger.info('training time this epoch: {}'.format(epoch_time))
        # eval
        eval_loss, eval_acc = eval_nsp(CPCmodel, encoding_model, loader_eval, device, epoch, args, logger, args.calculate_test_acc)
        print('Eval_loss: {:.4f}'.format(eval_loss))
        logger.info('Eval_loss: {:.4f}'.format(eval_loss))
        if eval_acc != -1:
            print('eval acc: ',eval_acc)
            logger.info('Eval_acc: {:.4f}'.format(eval_acc))
        

        
        # Save the model if the validation loss is the best we've seen so far.
        if  eval_loss < best_eval_loss:
            best_model_epoch = epoch
            if not args.trial or args.trial_save_model:
                if args.trial_save_model:
                    print('!!!!!!!! This is a test run, remove --trial_save_model in actual run!!!!!!')
                
                now = datetime.datetime.now()
                curr_date_time = now.strftime("%Y-%m-%d")
                    
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': CPCmodel.state_dict(),
                            'encoder_model_state_dict': encoding_model.state_dict(),
                            'loss_state_dict':loss_func.state_dict(),
                            'optimizer_state_dict': optim.state_dict(),
                            'loss': eval_loss,
                            },  args.save_model_dir +curr_date_time + str(args.save_model_name)+'-nsp-best_model'+'.pt')
                #torch.save(fbmodel.state_dict(), args.save_model_dir +curr_date + str(args.model_type)+\
                #           'FB='+str(args.FB_function_size) +'_lr='+str(args.lr)+'_model''.epoch_{}'.format(epoch)+'.pt')
            best_eval_loss = eval_loss
            n_plateau = 0
            
        elif n_plateau >2 and not args.use_cpc_nlp_optim:
            # Anneal the learning rate if no improvement has been seen in the
            # validation dataset.
            lr /= lr_decay
            print('lr decayed to: ', lr)
            n_plateau = 0
        else:
            n_plateau +=1
        
        ## report current best model:
        print('Current best model is from epoch {} with eval loss {:.4f}'.format(best_model_epoch, best_eval_loss))
        logger.info('Current best model is from epoch {} with eval loss {:.4f}'.format(best_model_epoch, best_eval_loss))

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