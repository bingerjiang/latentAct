#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=7
# python simcse_visualize.py --pretrained \
#                     --model 'simcse' \
#                     --tsne_plot_dir '../simcse_plots/' \
#                     --pretrained_model_name "princeton-nlp/sup-simcse-bert-base-uncased" \
#                     --dataset 'daily_dialog' \
#                     --save_csv \
#                     --vector_quantize \
#                     --make_plot \
#                     --csv_dir './sent_csvs/simcse/'\
export CUDA_VISIBLE_DEVICES=7
python simcse_visualize.py --pretrained \
                    --model 'cpc' \
                    --tsne_plot_dir '../cpc_plots/' \
                    --pretrained_model_name "../cpc_models/2022-03-28-best_model.epoch_7.pt" \
                    --dataset 'meta_woz' \
                    --save_csv \
                    --make_plot \
                    --csv_dir '../cpc_csvs/'\
