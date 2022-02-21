# export CUDA_VISIBLE_DEVICES=3
# python train_with_args.py --k_neg_pos 10 \
#                       --n_epochs 1\
#                       --eval_batch_size 4\
#                       --train_batch_size 1\
#                       --model_type 'cos_tlayer'\
#                       --lr 1e-5\
#                       --save_model_dir '../cos_tlayer_model/'\
#                       --flex\
#                       --FB_function_size 60\

# 0211
# export CUDA_VISIBLE_DEVICES=3
# python train_with_args.py --k_neg_pos 10 \
#                       --n_epochs 100\
#                       --eval_batch_size 4\
#                       --train_batch_size 1\
#                       --model_type 'cos'\
#                       --lr 1e-5\
#                       --save_model_dir '../cos_state_dicts/'\
#                       --flex\
#                       --FB_function_size 60\

# export CUDA_VISIBLE_DEVICES=4
# python train_with_args.py --k_neg_pos 10 \
#                       --n_epochs 100\
#                       --eval_batch_size 4\
#                       --train_batch_size 1\
#                       --model_type 'cos'\
#                       --lr 5e-5\
#                       --save_model_dir '../cos_state_dicts/'\
#                       --flex\
#                       --FB_function_size 60\

# 0213 train 2 binary 2 costlayer
# export CUDA_VISIBLE_DEVICES=3
# python train_with_args.py --k_neg_pos 10 \
#                       --n_epochs 100\
#                       --eval_batch_size 4\
#                       --train_batch_size 1\
#                       --model_type 'binary'\
#                       --lr 1e-5\
#                       --save_model_dir '../binary_state_dicts/'\
#                       --flex\
#                       --FB_function_size 4\

# export CUDA_VISIBLE_DEVICES=4
# python train_with_args.py --k_neg_pos 10 \
#                       --n_epochs 100\
#                       --eval_batch_size 4\
#                       --train_batch_size 1\
#                       --model_type 'binary'\
#                       --lr 5e-5\
#                       --save_model_dir '../binary_state_dicts/'\
#                       --flex\
#                       --FB_function_size 4\
# export CUDA_VISIBLE_DEVICES=5
# python train_with_args.py --k_neg_pos 10 \
#                       --n_epochs 100\
#                       --eval_batch_size 4\
#                       --train_batch_size 1\
#                       --model_type 'cos_tlayer'\
#                       --lr 1e-5\
#                       --save_model_dir '../costlayer_state_dicts/'\
#                       --flex\
#                       --FB_function_size 4\           
# export CUDA_VISIBLE_DEVICES=7
# python train_with_args.py --k_neg_pos 10 \
#                       --n_epochs 100\
#                       --eval_batch_size 4\
#                       --train_batch_size 1\
#                       --model_type 'cos_tlayer'\
#                       --lr 5e-5\
#                       --save_model_dir '../costlayer_state_dicts/'\
#                       --flex\
#                       --FB_function_size 4\
# export CUDA_VISIBLE_DEVICES=5
# python train_with_args.py --k_neg_pos 10 \
#                       --n_epochs 30\
#                       --eval_batch_size 4\
#                       --train_batch_size 1\
#                       --model_type 'binary'\
#                       --lr 1e-5\
#                       --save_model_dir '../binary_state_dicts/'\
#                       --flex\
#                       --FB_function_size 32\
#                       --pre_z_tanh

# export CUDA_VISIBLE_DEVICES=7
# python train_with_args.py --k_neg_pos 10 \
#                       --n_epochs 30\
#                       --eval_batch_size 4\
#                       --train_batch_size 1\
#                       --model_type 'binary'\
#                       --lr 5e-5\
#                       --save_model_dir '../binary_state_dicts/'\
#                       --flex\
#                       --FB_function_size 32\        
#                       --pre_z_tanh 

# export CUDA_VISIBLE_DEVICES=3
# python train_with_args.py --k_neg_pos 10 \
#                       --n_epochs 30\
#                       --eval_batch_size 4\
#                       --train_batch_size 1\
#                       --model_type 'cos_tlayer'\
#                       --lr 1e-5\
#                       --save_model_dir '../costlayer_state_dicts/'\
#                       --flex \
#                       --FB_function_size 32 \
#                       --pre_z_tanh 

export CUDA_VISIBLE_DEVICES=4
python train_with_args.py --k_neg_pos 10 \
                      --n_epochs 30\
                      --eval_batch_size 4\
                      --train_batch_size 1\
                      --model_type 'cos_tlayer'\
                      --lr 5e-5\
                      --save_model_dir '../costlayer_state_dicts/'\
                      --flex\
                      --FB_function_size 32\        
                      --pre_z_tanh 