#!/usr/bin/env bash

## note: if plot, don't include negatives --eval_sample_negatives
##       otherwise it will mess up the representation


# export CUDA_VISIBLE_DEVICES=2
# python visualize.py --load_model_name '2022-02-14binaryFB=4_lr=5e-05_model.epoch_1.pt' \
#                --model_dir  '../binary_state_dicts/'\
#                --tsne_plot_dir '../tsne_binary_4/'\
#                --csv_dir './sent_csvs/'\
#                --FB_function_size 4\
#                --load_state_dict_new\
#                --model_type 'binary' \
#                --make_plot \
#                --plot_kmeans_tsne\
#                --save_csv

# export CUDA_VISIBLE_DEVICES=2
# python visualize.py --load_model_name '2022-02-15cos_tlayerFB=4_lr=5e-05_model.epoch_5.pt' \
#                --model_dir  '../costlayer_state_dicts/'\
#                --tsne_plot_dir '../tsne_cost_4/'\
#                --csv_dir './csvs/'\
#                --FB_function_size 4\
#                --load_state_dict_new\
#                --model_type 'cos_tlayer' \
#                --make_plot \
#                --plot_kmeans_tsne\
#                --do_tsne

# loss 0.36               


# 'lr=1e-5by5_model.epoch_0'

# k=3； can't be larger than 4, CUDA out of memory
# 'size=64_lr=1e-5b_model.epoch_1'
#96.96% 96.57%(k=4)
# './model_checkpoints/lr=1e-4by5_model.epoch_0'
# 75%
# lr=1e-5by5_model.epoch_0
# 96.95% (k=3) 97.89% (k=1) 96.82%（k=4)

# python visualize.py --load_model_name 'lr=5e-6_model.epoch_3' \
#                --model_dir  './old/model_checkpoints/'\
#                --tsne_plot_dir './tsne_768/'\
#                --csv_dir './csvs/'\
#                -k 2\
#                --FB_function_size 768\
#                --do_tsne\
# export CUDA_VISIBLE_DEVICES=2
# python visualize.py --load_model_name '2022-02-14binaryFB=4_lr=5e-05_model.epoch_1.pt' \
#                --model_dir  '../binary_state_dicts/'\
#                --tsne_plot_dir '../tsne_binary_32/'\
#                --csv_dir './sent_csvs/'\
#                --FB_function_size 32\
#                --load_state_dict_new\
#                --model_type 'binary' \
#                --make_plot \
#                --plot_kmeans_tsne\
#                --save_csv
# export CUDA_VISIBLE_DEVICES=2
# python visualize.py --load_model_name '2022-02-22binaryFB=32_lr=1e-05_model.epoch_2.pt' \
#                --model_dir  '../binary_state_dicts/'\
#                --tsne_plot_dir '../tsne_binary_32/'\
#                --csv_dir './sent_csvs/'\
#                --FB_function_size 32\
#                --load_state_dict_new\
#                --model_type 'binary' \
#                --make_plot \
#                --plot_kmeans_tsne\
#                --save_csv \
#                --pre_z_tanh

# export CUDA_VISIBLE_DEVICES=1
# python visualize.py --load_model_name '2022-02-22cos_tlayerFB=32_lr=1e-05_model.epoch_3.pt' \
#                --model_dir  '../costlayer_state_dicts/'\
#                --tsne_plot_dir '../tsne_cost_32/'\
#                --csv_dir './sent_csvs/'\
#                --FB_function_size 32\
#                --load_state_dict_new\
#                --model_type 'cos_tlayer' \
#                --make_plot \
#                --plot_kmeans_tsne\
#                --save_csv \
#                --pre_z_tanh

# export CUDA_VISIBLE_DEVICES=7
# python visualize.py --load_model_name '2022-02-22binaryFB=32_lr=1e-05_model.epoch_2.pt' \
#                --model_dir  '../binary_state_dicts/'\
#                --tsne_plot_dir '../metawoz_plots/tsne_metawoz_binary32/'\
#                --csv_dir './sent_csvs/vq/'\
#                --FB_function_size 32\
#                --load_state_dict_new\
#                --model_type 'binary' \
#                --calculate_accuracy\
#                --pre_z_tanh \
#                --dataset 'daily_dialog'\
#                --eval_sample_negatives \
#                --test_k 1\
#                --vector_quantize \
#                --save_csv \
#                --make_plot \
               #--plot_kmeans_tsne\
               #--save_csv \

# for testing simcse
# ATTENTION!!! NO --eval_sample_negatives for visualization!!!
export CUDA_VISIBLE_DEVICES=7
python visualize.py --load_model_name '2022-02-22binaryFB=32_lr=1e-05_model.epoch_2.pt' \
               --model_dir  '../binary_state_dicts/'\
               --tsne_plot_dir '../metawoz_plots/tsne_metawoz_binary32/'\
               --csv_dir './sent_csvs/vq/'\
               --FB_function_size 32\
               --load_state_dict_new\
               --model_type 'binary' \
               --calculate_accuracy\
               --pre_z_tanh \
               --dataset 'daily_dialog'\
               --vector_quantize \
               --save_csv \
               --make_plot \


