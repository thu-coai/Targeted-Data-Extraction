file_path=../datasets/train_prefix.npy
# file_path=./data/valid_prefix.npy
CUDA_VISIBLE_DEVICES=2 python gen.py --root-dir res/ --experiment-name genseed2022_trainseed42_token100_maxlossToken5_alpha0.7_lr1e-3_warmup500_1gpu_maxepoch20_half_topp0.7_temp0.8_trial100/ --num-trials 100 \
    --guess-prefix=guess --file-path=${file_path} \
    --prefix-len=50 --suffix-len=50 --chunk=1000 --bs=64 --seed=2022

# CUDA_VISIBLE_DEVICES=2 python gen.py --root-dir res/ --experiment-name token100_maxepoch20_half_topp0.7_add25gentoken_trial100/ --num-trials 100 \
#     --guess-prefix=guess --dataset-dir=../gen_dataset/token100_topp0.7_guess100_add25 \
#     --prefix-len=75 --suffix-len=25 --chunk=0 --bs=64