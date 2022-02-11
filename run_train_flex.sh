export CUDA_VISIBLE_DEVICES=5
python train_with_args.py --k_neg_pos 10 \
                      --n_epochs 3\
                      --eval_batch_size 4\
                      --train_batch_size 1\
                      --model_type 'cos_tlayer'\
                      --lr 1e-5\
                      --save_model_dir '../cos_tlayer_model/'\
                      --trial\
                      --flex\
                      --FB_function_size 60\


                     
                      #--continue_train \
                      #--load_model_name 'lr=1e-5_model.epoch_3'\