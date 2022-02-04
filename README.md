# latentAct

progress:

training 4 with different lr on DeepSleep 

train_with_args.py: train_cuda2.py on DeepSleep, same script. (fixed negative sampling issue, now works for all batch sizes and k's; train_batch_size can't be larger than 2 for CUDA memory, even for pbody.)
