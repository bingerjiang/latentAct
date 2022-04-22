
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
from pytorch_metric_learning.distances import CosineSimilarity

from tools import *
from models import *
from dataset import *
from evaluation import *
from test import *
from fb_transformer import *
#%%
#logger = setup_logger('{}'.format('model.pt'))
import datetime
import argparse
from torch_lr_finder import LRFinder
def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
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
        model_params = list(CPCmodel.parameters()) + list(encoding_model.parameters())
    
    if args.use_cpc_nlp_optim:
        optim = torch.optim.Adam(model_params, 
            lr=2e-4, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True)
    else:
        optim = AdamW(model_params, lr=lr)
    
    loss_func = NTXentLoss()

    model_path = args.load_model_dir
    
    best_eval_loss = 100000
    best_model_epoch = -1
    
    n_plateau= 0
    epochs = args.n_epochs 
    k =args.k_neg_pos
    CPCmodel.to(device)

if __name__ == "__main__":
    main()