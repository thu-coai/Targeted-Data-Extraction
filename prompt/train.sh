data_dir=data
max_epoch=20
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 \
    --master_port 29584 \
    main.py --train_prefix_path=./${data_dir}/train_prefix.npy \
    --val_prefix_path=./${data_dir}/valid_prefix.npy \
    --train_suffix_path=./${data_dir}/train_suffix.npy \
    --val_suffix_path=./${data_dir}/valid_suffix.npy \
    --basemodel_path=/home/zhangzhexin/huggingface_pretrained_models/gpt-neo-1.3B \
    --model_type=softprompt \
    --token_num=100 \
    --need_loss_len=50 \
    --useneg=0 --neg_loss=contrastive \
    --margin=0 --temperature=1.0 \
    --maxloss_tokennum=5 \
    --minloss=0 \
    --extraloss_alpha=0.7 \
    --train_negsuffix_path=./${data_dir}/train_neg_pred.npy \
    --val_negsuffix_path=./${data_dir}/valid_neg_pred.npy \
    --pretrained_model_path= \
    --batch_size=16 --val_batch_size=16 --num_workers=4 \
    --gradient_accumulation_steps=2 \
    --seed=42 \
    --fp16= \
    --savedmodel_path=./save/token100_maxlossToken5_alpha0.7_needlosslen50/seed42_realbs32_1gpu_lr1e-3_warmup500_lineardecay_maxepoch${max_epoch} --ckpt_file='' \
    --max_epochs=${max_epoch} --warmup_steps=500 --warmup_ratio=0 \
    --learning_rate=1e-3 \
    --lr_decay=linear --patience=5 \
    --ema=0 --ema_start_epoch=0
    
