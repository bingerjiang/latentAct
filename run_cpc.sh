#!/usr/bin/env bash

# 0328 run #1
# export CUDA_VISIBLE_DEVICES=6
# python cpc_train.py  --dataset 'meta_woz'\
#                      --use_cpc_nlp_optim \
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 20 \
#                      --save_model_name '03-28_vanila_w_cpc-nlp-optim'

## 0328 run #2
## differ only in optim from above
# export CUDA_VISIBLE_DEVICES=6
# python cpc_train.py  --dataset 'meta_woz' \
#                      --save_model_dir '../cpc_models/' \
#                      --save_model_name '03-28_vanila_w_AdamW'\
#                      --n_epochs 20 \
                     
## 0328 run #3
## unfreeze sentence bert; size 7738MB
# export CUDA_VISIBLE_DEVICES=5
# python cpc_train.py  --dataset 'meta_woz'\
#                      --use_cpc_nlp_optim \
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 20 \
#                      --save_model_name '03-28_unfreeze_sbert'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\

# test trial run
# export CUDA_VISIBLE_DEVICES=3
# python cpc_train.py  --dataset 'meta_woz'\
#                      --use_cpc_nlp_optim \
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 1 \
#                      --save_model_name 'testcode'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --trial  \
#                      --sent_encoding_model 'simcse'


# 0401 run #1
# export CUDA_VISIBLE_DEVICES=5
# python cpc_train.py  --dataset 'meta_woz'\
#                      --use_cpc_nlp_optim \
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 30 \
#                      --save_model_name 'unfreeze_sbert_nlpOsptim'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'sbert'

# 0401 run #2 √
# export CUDA_VISIBLE_DEVICES=4
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 30 \
#                      --save_model_name 'unfreeze_sbert_adamW'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'sbert'

# 0401 run #3 simcse √
# export CUDA_VISIBLE_DEVICES=6
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 30 \
#                      --save_model_name 'unfreeze_simcse_adamW'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'

# 0401 run #4 simcse
# export CUDA_VISIBLE_DEVICES=7
# python cpc_train.py  --dataset 'meta_woz'\
#                      --use_cpc_nlp_optim \
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 30 \
#                      --save_model_name 'unfreeze_simcse_nlpOsptim'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'

#0401 run #2  continue
# export CUDA_VISIBLE_DEVICES=5
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 50 \
#                      --save_model_name 'unfreeze_sbert_adamW_continue'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'sbert'\
#                      --continue_train\
#                      --load_model_dir '../cpc_models/'\
#                      --load_model_name '2022-04-02unfreeze_sbert_adamW'\
                     

# 0401 run #3 simcse continue
# export CUDA_VISIBLE_DEVICES=6
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 50 \
#                      --save_model_name 'unfreeze_simcse_adamW_continue'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'\
#                      --continue_train\
#                      --load_model_dir '../cpc_models/'\
#                      --load_model_name '2022-04-02unfreeze_simcse_adamW'\

# 0404 run #changelr simcse 
# export CUDA_VISIBLE_DEVICES=7
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 30 \
#                      --save_model_name 'unfreeze_simcse_adamW_1e-5'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'\
#                      --lr 1e-5            

# 0404 run #changelr2 simcse
# export CUDA_VISIBLE_DEVICES=3
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 30 \
#                      --save_model_name 'unfreeze_simcse_adamW_5e-6'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'\
#                      --lr 5e-6             

## 0404 transformer (but found issues in parameters 0405 -- not updating encoding_model)
# export CUDA_VISIBLE_DEVICES=4
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 30 \
#                      --save_model_name 'transformer'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'\
#                      --lr 1e-5\
#                      --CPC_AR_model_type 'transformer'\

## 0405 transformer fixed (now training sentence encoding model)
# export CUDA_VISIBLE_DEVICES=5
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 30 \
#                      --save_model_name 'transformer_fixed'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'\
#                      --lr 1e-5\
#                      --CPC_AR_model_type 'transformer'\

## 0405 2 transformer layers
# export CUDA_VISIBLE_DEVICES=7
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 30 \
#                      --save_model_name 'transformer_2layer'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'\
#                      --lr 1e-5\
#                      --CPC_AR_model_type 'transformer'\
#                      --n_transformer_layer 2

## same with above, but having cos accuracy
# export CUDA_VISIBLE_DEVICES=4
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 30 \
#                      --save_model_name 'transformer_2layer-add_cos_accuracy'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'\
#                      --lr 1e-5\
#                      --CPC_AR_model_type 'transformer'\
#                      --n_transformer_layer 2\

# export CUDA_VISIBLE_DEVICES=6
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 30 \
#                      --save_model_name 'transformer_1layer-add_cos_accuracy'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'\
#                      --lr 1e-5\
#                      --CPC_AR_model_type 'transformer'\
#                      --n_transformer_layer 1\

## can't load model 2layers -- out of memory
# export CUDA_VISIBLE_DEVICES=2
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 1 \
#                      --save_model_name 'test-accuracy'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'\
#                      --lr 1e-5\
#                      --CPC_AR_model_type 'transformer'\
#                      --n_transformer_layer 2\
#                      --continue_train\
#                      --load_model_dir '../cpc_models/'\
#                      --load_model_name '2022-04-05transformer_2layer-add_cos_accuracy'\

## 0406 fixed cos sim accuracy bugs, now training on transformer with lr=1e-4
# export CUDA_VISIBLE_DEVICES=4
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 50 \
#                      --save_model_name 'transformer-1e-4'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'\
#                      --lr 1e-4\
#                      --CPC_AR_model_type 'transformer'\
#                      --n_transformer_layer 1\

## 0406 fixed cos sim accuracy bugs, now training on transformer with lr=5e-5
# export CUDA_VISIBLE_DEVICES=7
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 50 \
#                      --save_model_name 'transformer-5e-5'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'\
#                      --lr 5e-5\
#                      --CPC_AR_model_type 'transformer'\
#                      --n_transformer_layer 1\

# export CUDA_VISIBLE_DEVICES=6
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 100 \
#                      --save_model_name 'transformer-1e-5'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'\
#                      --lr 1e-5\
#                      --CPC_AR_model_type 'transformer'\
#                      --n_transformer_layer 1\

# export CUDA_VISIBLE_DEVICES=6
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 1 \
#                      --save_model_name 'testdataset'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'\
#                      --lr 1e-5\
#                      --CPC_AR_model_type 'transformer'\
#                      --n_transformer_layer 1\

## 0422 shuffled dataset so eval is all included in train, lr = 1e-5
# export CUDA_VISIBLE_DEVICES=3
# python cpc_train.py  --dataset 'meta_woz'\
#                      --save_model_dir '../cpc_models/'\
#                      --n_epochs 100 \
#                      --save_model_name 'transformer_shuffled_1e-5'\
#                      --finetune_sent_encoding\
#                      --n_turns 11\
#                      --batch_size 6\
#                      --sent_encoding_model 'simcse'\
#                      --lr 1e-5\
#                      --CPC_AR_model_type 'transformer'\
#                      --n_transformer_layer 1\

## 0422 shuffled dataset so eval is all included in train, lr = 5e-5
export CUDA_VISIBLE_DEVICES=7
python cpc_train.py  --dataset 'meta_woz'\
                     --save_model_dir '../cpc_models/'\
                     --n_epochs 100 \
                     --save_model_name 'transformer_shuffled_5e-5'\
                     --finetune_sent_encoding\
                     --n_turns 11\
                     --batch_size 6\
                     --sent_encoding_model 'simcse'\
                     --lr 5e-5\
                     --CPC_AR_model_type 'transformer'\
                     --n_transformer_layer 1\