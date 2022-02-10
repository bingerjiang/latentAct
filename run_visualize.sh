#!/usr/bin/env bash

## note: if plot, don't include negatives --eval_sample_negatives
##       otherwise it will mess up the representation
export CUDA_VISIBLE_DEVICES=5
python visualize.py --load_model_name 'lr=5e-6_model.epoch_3' \
               --model_dir  './old/model_checkpoints/'\
               --tsne_plot_dir './tsne_768/'\
               --csv_dir './csvs/'\
               -k 2\
               --FB_function_size 768\
               --do_tsne\

               

# 'lr=1e-5by5_model.epoch_0'

# k=3； can't be larger than 4, CUDA out of memory
# 'size=64_lr=1e-5b_model.epoch_1'
#96.96% 96.57%(k=4)
# './model_checkpoints/lr=1e-4by5_model.epoch_0'
# 75%
# lr=1e-5by5_model.epoch_0
# 96.95% (k=3) 97.89% (k=1) 96.82%（k=4)