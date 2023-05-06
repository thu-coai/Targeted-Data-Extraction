# Copyright 2022 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from asyncio import ensure_future
from copy import deepcopy
from functools import partial
from random import sample
import json
import numpy as np
import csv
import transformers
import zlib
import os
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import roc_auc_score

tokenizer = transformers.GPT2Tokenizer.from_pretrained("/home/zhangzhexin/huggingface_pretrained_models/gpt2")

def generate_full_curve(rows, num_samples):
    did_solve = np.zeros(num_samples)
    
    recall = []
    errors = []

    bad_guesses = 0

    answer = None

    for exid, is_correct in rows:
        if is_correct:
            did_solve[int(exid)] = 1
            recall.append(np.mean(did_solve))
            errors.append(bad_guesses)
            if bad_guesses < 100:
                answer = np.mean(did_solve)
        else:
            bad_guesses += 1

    print("Recall at 100 errors", answer)
            
    try:
        import matplotlib.pyplot as plt
        plt.plot(errors, recall)

        plt.semilogx()
        plt.xlabel("Number of bad guesses")
        plt.ylabel("Recall")
        
        plt.savefig("/tmp/error_curve.png")
        print("A full error curve is located at /tmp/error_curve.png")
        
    except:
        print("Can't generate error curve; please install matplotlib to see the plot")

    return recall, errors
    
answer_path = './datasets/train_suffix.npy'
prefix_path = './datasets/train_prefix.npy'
sample_num = 1000

if sample_num is not None:
    answers = np.load(answer_path)[-sample_num:]
    prefixs = np.load(prefix_path)[-sample_num:]
else:
    answers = np.load(answer_path)
    prefixs = np.load(prefix_path)
    sample_num = len(answers)

compute_using_num = 25 # sample time

preds = [[] for _ in range(sample_num)]
losses = [[] for _ in range(sample_num)]
all_token_losses = [[] for _ in range(sample_num)]
compare_losses =  [[] for _ in range(sample_num)]

strresult_path = './result/res.json'
os.makedirs('./result', exist_ok=True)

_PREFIX_LEN = 50
_SUFFIX_LEN = 50
equal_compute_len = 50 # 只看生成后缀的前多少个token

comparedir = 'gpt2loss'
docompare = False
use_all_token_loss = False
conf_type = 'ours'
# conf_type = 'ppl'
# conf_type = 'compare'

for i in range(compute_using_num):
    
    resdir = f'./prompt/res/genseed2022_trainseed1000_token100_maxlossToken5_alpha0.7_lr1e-3_warmup500_1gpu_maxepoch20_half_topp0.7_temp0.8_trial100'
    a = np.load(f'{resdir}/generations/{i}.npy')
    b = np.load(f'{resdir}/losses/{i}.npy').reshape(-1)
    if use_all_token_loss:
        all_token_loss = np.load(f'{resdir}/all_token_losses/{i}.npy')
    if docompare:
        c = np.load(f'{resdir}/{comparedir}/{i}.npy').reshape(-1)
    
    alpha = 1.0
    for j in range(len(a)):
        preds[j].append(a[j])
        # print(b[j], c[j])
        # zlib_score = len(zlib.compress(bytes(tokenizer.decode(a[j][-_SUFFIX_LEN:]), encoding='utf-8')))
        # losses[j].append(float(b[j]) / zlib_score)
        losses[j].append(b[j])
        if use_all_token_loss:
            all_token_losses[j].append(all_token_loss[j])
        if docompare:
            compare_losses[j].append(c[j])
            # compare_losses[j].append(zlib_score)

def get_neg_confidence(item=None, x=None):
    # item: one pred, x: all_preds
    # 越小代表概率越大
    # print(x)
    if item is None:
        # 整体计算，更加高效
        # exp_seq = np.array([np.exp(-t['loss']) for t in x])
        # total_prob = exp_seq.sum()
        # each_prob = exp_seq / total_prob
        counter = Counter()
        pred2id = {}
        nodup_loss_seq = []
        idxmap = {}
        for i, t in enumerate(x):
            if tuple(t['pred']) not in pred2id:
                nodup_loss_seq.append(t['loss'])
                idxmap[i] = len(nodup_loss_seq) - 1
                pred2id[tuple(t['pred'])] = idxmap[i]
            else:
                idxmap[i] = pred2id[tuple(t['pred'])]

            # counter[tuple(t['pred'])] += 1
            counter[idxmap[i]] += 1
        
        # nodup_exp_seq = np.exp(-np.array(nodup_loss_seq))
        
        weights_seq = []
        for i in range(len(nodup_loss_seq)):
            if counter[i] <= 0:
                weights_seq.append(1)
            else:
                weights_seq.append(counter[i] ** 1.0)
        # print(weights_seq)
        nodup_loss_seq = np.array(nodup_loss_seq)
        # nodup_loss_seq = -np.array(nodup_loss_seq) * np.array(weights_seq)
        # nodup_exp_seq = np.exp(-nodup_loss_seq)
        nodup_exp_seq = np.exp(-nodup_loss_seq * equal_compute_len)

        nodup_exp_seq *= np.array(weights_seq)
        nodup_total_prob = nodup_exp_seq.sum()
        nodup_each_prob = nodup_exp_seq / nodup_total_prob
        
        # print(idxmap, nodup_each_prob)
        for i, t in enumerate(x):
            if conf_type == 'compare':
                t['confidence'] = float(t['loss'] / t['compareloss'])
            elif conf_type == 'ppl':
                t['confidence'] = float(t['loss'])
            elif conf_type == 'ours':
                t['confidence'] = float(-nodup_each_prob[idxmap[i]])

    else:
    
        total_prob = np.sum([np.exp(-t['loss']) for t in x])
        rank0_prob = np.sum([np.exp(-t['loss']) for t in x if np.all(t['pred'] == item['pred'])])
        if docompare:
            compare_total_prob = np.sum([np.exp(-t['compareloss']) for t in x])
            compare_rank0_prob = np.sum([np.exp(-t['compareloss']) for t in x if np.all(t['pred'] == item['pred'])])
            return -compare_rank0_prob / compare_total_prob
            
        return -rank0_prob / total_prob
       
res = []
loss_accs_preds = []
for i in tqdm(range(len(preds))):
    answer = answers[i]
    temp_preds = preds[i]
    temp_losses = np.array(losses[i])
    
    temp_list = []
    str_temp_list = []
    str_acc = 0
    if docompare:
        temp_compare_losses = np.array(compare_losses[i])
    for idx, (pred, loss) in enumerate(zip(temp_preds, temp_losses)):
        if not equal_compute_len == _SUFFIX_LEN:
            acc = np.all(pred[-_SUFFIX_LEN:-_SUFFIX_LEN+equal_compute_len] == answer[-_SUFFIX_LEN:-_SUFFIX_LEN+equal_compute_len])
        else:
            acc = np.all(pred[-_SUFFIX_LEN:] == answer[-_SUFFIX_LEN:])
        if docompare:
            compare_loss = temp_compare_losses[idx]
            temp_list.append({'loss': loss, 'acc': acc, 'pred': pred, 'compareloss': compare_loss})
        else:
            temp_list.append({'loss': loss, 'acc': acc, 'pred': pred})
        equal = int(acc)
        # print(len(tokenizer))
        # print(pred[-_SUFFIX_LEN:])
        if use_all_token_loss:
            all_token_loss = all_token_losses[i][idx]
            str_temp_list.append({'pred': tokenizer.decode(pred[-_SUFFIX_LEN:], skip_special_tokens=False),
                            'loss': float(loss), 'all_token_loss':all_token_loss.tolist(), 'correct': equal, 'pred_ids': pred[-_SUFFIX_LEN:].tolist()})
        else:
            str_temp_list.append({'pred': tokenizer.decode(pred[-_SUFFIX_LEN:], skip_special_tokens=False),
                            'loss': float(loss), 'correct': equal, 'pred_ids': pred[-_SUFFIX_LEN:].tolist()})
        
        if equal:
            str_acc = 1
        
    # temp_list.sort(key=lambda item:item['loss'])
    # for i, sample in enumerate(temp_list):
    #     sample['confidence'] = get_neg_confidence(sample, temp_list)
    #     str_temp_list[i]['confidence'] = float(sample['confidence'])
    get_neg_confidence(x=temp_list)
    for q in range(len(str_temp_list)):
        str_temp_list[q]['confidence'] = temp_list[q]['confidence']
        
    temp_list.sort(key=lambda x:x['confidence'])
    assert temp_list
    loss_accs_preds.append(temp_list)
    
    suffix = tokenizer.decode(answer[-_SUFFIX_LEN:])
    str_temp_list.sort(key=lambda x:x['confidence'])
    # 合并重复的预测
    new_str_temp_list = []
    start = 0
    while start < len(str_temp_list):
        move = start + 1
        repeat_time = 1
        while move < len(str_temp_list):
            if str_temp_list[move]['pred'] == str_temp_list[start]['pred']:
                repeat_time += 1
                move += 1
            else:
                break
        new_item = str_temp_list[start]
        new_item['repeat_time'] = repeat_time
        new_str_temp_list.append(new_item)
        start = move
        
    prefix = tokenizer.decode(prefixs[i])
    temp_dict = {
        'prefix': prefix,
        'prefix_ids': prefixs[i].tolist(),
        'answer': suffix,
        'answer_ids': answer[-_SUFFIX_LEN:].tolist(),
        'correct': str_acc,
        'preds': new_str_temp_list
    }
    res.append(temp_dict)

with open(strresult_path, 'w') as outf:
    json.dump(res, outf, ensure_ascii=False, indent=1)

loss_accs_preds.sort(key=lambda x:x[0]['confidence'])

mean_rank0_loss = np.mean([x[0]['loss'] for x in loss_accs_preds])
true_losses = [x[0]['loss'] for x in loss_accs_preds if x[0]['acc']]

wrong_cnt = 0
true_cnt = 0
for x in loss_accs_preds:
    if x[0]['acc']:
        true_cnt += 1
    else:
        wrong_cnt += 1
    
    if wrong_cnt == 100:
        break
    
print(f'Recall (early stop): {true_cnt / 1000}')


ranks = []
for x in loss_accs_preds:
    acc = 0
    for i, item in enumerate(x):
        if item['acc']:
            acc = 1
            ranks.append(i)
            break
    if not acc:
        ranks.append(compute_using_num)
        
ranks = np.array(ranks)
recall_1 = (ranks == 0).sum() / sample_num
print(f'Recall: {recall_1}')

