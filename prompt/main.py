import logging
import os
import time
import numpy as np

import torch
import torch.nn.functional as F

from config import parse_args
from pytorch_lightning import seed_everything
from data_helper import create_dataloaders
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM, AutoModelForSequenceClassification
from accelerate import Accelerator, DistributedDataParallelKwargs
from util import setup_logging, build_optimizer, EMA
from model import SoftPromptModel, PrefixModel

from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR
import datasets

logging.disable(logging.WARNING)
# import warnings
# warnings.filterwarnings("ignore")


# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

def validate(model, val_dataloader, accelerator, args):
    model.eval()
    
    losses = []
    # accs = []
    extra_losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, total=len(val_dataloader)):
            
            if not args.useneg:
                input_ids = batch['input_ids']
                outputs = model(input_ids)
                # print(f'outputs keys:{outputs.keys()}')
                loss = outputs.loss
            else:
                input_ids = batch['input_ids']
                outputs = model(input_ids)
                neg_input_ids = batch['neg_input_ids']
                neg_outputs = model(neg_input_ids)
                if args.neg_loss == 'margin':
                    mask = batch['mask']
                    pos_loss = torch.masked_select(outputs.each_loss, mask)
                    neg_loss = torch.masked_select(neg_outputs.each_loss, mask)
                    margin_fc = torch.nn.MarginRankingLoss(margin=args.margin)
                    target = torch.ones_like(pos_loss)
                    margin_loss = margin_fc(neg_loss, pos_loss, target)
                    loss = outputs.loss + margin_loss
                elif args.neg_loss == 'contrastive':
                    mask = batch['mask']
                    pos_loss = torch.masked_select(outputs.each_loss, mask)
                    neg_loss = torch.masked_select(neg_outputs.each_loss, mask)
                    
                    stack_loss = -torch.stack([pos_loss, neg_loss], dim=1)
                    log_softmax_loss = -F.log_softmax(stack_loss, dim=-1)[:, 0]
                    contrastive_loss = log_softmax_loss.mean()
                    loss = outputs.loss + contrastive_loss
                    
            
            losses.append(accelerator.gather(loss).mean().item())
            if args.maxloss_tokennum:
                # print(f'outputs.maxloss:{outputs.maxloss}')
                extra_losses.append(accelerator.gather(outputs.maxloss).mean().item())
            elif args.minloss:
                extra_losses.append(accelerator.gather(outputs.minloss).mean().item())
            elif args.useneg:
                if args.neg_loss == 'margin':
                    extra_losses.append(accelerator.gather(margin_loss).mean().item())
                elif args.neg_loss == 'contrastive':
                    extra_losses.append(accelerator.gather(contrastive_loss).mean().item())

            
    model.train()

    loss = sum(losses) / len(losses)
    if extra_losses:
        # margin_loss = sum(margin_losses) / len(margin_losses)
        extra_loss = sum(extra_losses) / len(extra_losses)
        return {'loss': loss, 'extra_loss': extra_loss}
    else:
        return {'loss': loss}
    


def train_and_validate(args):
    # 1. load data
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    mixed_precision = 'fp16' if args.fp16 else 'no'
    accelerator = Accelerator(mixed_precision=mixed_precision, kwargs_handlers=[ddp_kwargs])
    train_dataloader, val_dataloader = create_dataloaders(args)
    
    # if args.load_pretrained:
    #     checkpoint = torch.load(args.pretrained_model_path, map_location='cpu')
    #     model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    #     print("load pretrained model")
    total_gpu_cnt = torch.distributed.get_world_size()
    print(f'total gpu cnt:{total_gpu_cnt}')
    args.max_steps = len(train_dataloader) // (args.gradient_accumulation_steps * total_gpu_cnt) * args.max_epochs 
    print(f'max steps:{args.max_steps}')
    
    # model = MyGptModel(args)
    if args.model_type == 'softprompt':
        model = SoftPromptModel(args)
    elif args.model_type == 'prefix':
        model = PrefixModel(args)
    # model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    # model.config.pad_token_id = 50256
    if args.pretrained_model_path:
        ckpt = torch.load(args.pretrained_model_path, map_location='cpu')
        # model.resize_token_embeddings(len(tokenizer) - args.class_num)
        model.load_state_dict(ckpt['model_state_dict'])
    
    # model.resize_token_embeddings(len(tokenizer))

    optimizer, scheduler = build_optimizer(args, model)
    
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)
    
    # 如果做了prepare就会有几个gpu就更新几次
    # scheduler = accelerator.prepare(scheduler)
    
    if args.ema:
        ema = EMA(model, decay=args.ema)
    # 3. training
    
    step = 0
    best_val_loss = 100
    best_val_acc = 0
    early_stop = 0
    
    for epoch in range(args.max_epochs):
        print(f'epoch {epoch}, current lr:{scheduler.get_last_lr()}')
        if epoch == args.ema_start_epoch and args.ema:
            ema.register()
        avg_loss = []
        
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        model.train()

        for batch_idx, batch in enumerate(bar):
            # print(f'batch_idx {batch_idx}, current lr:{scheduler.get_last_lr()}')
            if not args.useneg:
                input_ids = batch['input_ids']
                outputs = model(input_ids)
                loss = outputs.loss
            else:
                pass
            
            avg_loss.append(accelerator.gather(loss).mean().item())
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            
            if batch_idx % args.gradient_accumulation_steps == 0:
                # accelerator.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                if args.ema and epoch >= args.ema_start_epoch:
                    ema.update()
                    
                optimizer.zero_grad()
                scheduler.step()
            
            step += 1
            bar.set_postfix(Epoch=epoch, 
                            loss=np.mean(avg_loss[-100:]))
        if args.ema and epoch >= args.ema_start_epoch:
            ema.apply_shadow()
        loss_dict = validate(model, val_dataloader, accelerator, args)
        if args.ema and epoch >= args.ema_start_epoch:
            ema.restore()
            
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.5f}")

        # 5. save checkpoint
        loss = loss_dict['loss']
        val_loss = loss
        if val_loss < best_val_loss:
            early_stop = 0
            # best_val_loss = val_loss
            best_val_loss = val_loss
            if args.ema and epoch >= args.ema_start_epoch:
                ema.apply_shadow()
            state_dict = model.module.state_dict()
            if 'extra_loss' in loss_dict:
                extra_loss = loss_dict['extra_loss']
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'val_loss': val_loss}, f'{args.savedmodel_path}/{args.ckpt_file}_epoch{epoch}_valloss{val_loss:.5f}_extraloss{extra_loss:.5f}.bin')
            else:
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'val_loss': val_loss}, f'{args.savedmodel_path}/{args.ckpt_file}_epoch{epoch}_valloss{val_loss:.5f}.bin')
            if args.ema and epoch >= args.ema_start_epoch:
                ema.restore()
        else:
            early_stop += 1
            if args.ema and epoch >= args.ema_start_epoch:
                ema.apply_shadow()
            state_dict = model.module.state_dict()

            if 'extra_loss' in loss_dict:
                extra_loss = loss_dict['extra_loss']
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'val_loss': val_loss}, f'{args.savedmodel_path}/{args.ckpt_file}_epoch{epoch}_valloss{val_loss:.5f}_extraloss{extra_loss:.5f}.bin')
            else:
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'val_loss': val_loss}, f'{args.savedmodel_path}/{args.ckpt_file}_epoch{epoch}_valloss{val_loss:.5f}.bin')
                
            if args.ema and epoch >= args.ema_start_epoch:
                ema.restore()
            if early_stop >= args.patience:
                print(f"Early Stop for {args.savedmodel_path}")
                break
                
def main():
    args = parse_args()
    seed_everything(args.seed)
    setup_logging()

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)

if __name__ == '__main__':
    main()