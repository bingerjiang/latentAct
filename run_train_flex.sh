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

export CUDA_VISIBLE_DEVICES=5
python train_with_args.py --k_neg_pos 10 \
                      --n_epochs 1\
                      --eval_batch_size 4\
                      --train_batch_size 1\
                      --model_type 'cos'\
                      --lr 1e-5\
                      --save_model_dir '../cos_state_dicts/'\
                      --flex\
                      --FB_function_size 20\
                      --trial