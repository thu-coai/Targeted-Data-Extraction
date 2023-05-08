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

from absl import app
from absl import flags
from absl import logging
import csv
import os
import tempfile
from typing import Tuple, Union

import numpy as np
import transformers
import torch
from model import SoftPromptModel
from pytorch_lightning import seed_everything

from tqdm import tqdm

_ROOT_DIR = flags.DEFINE_string(
    'root-dir', "tmp/",
    "Path to where (even intermediate) results should be saved/loaded."
)
_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment-name',
    'sample',
    "Name of the experiment. This defines the subdir in `root_dir` where "
    "results are saved.")
_DATASET_DIR = flags.DEFINE_string(
    "dataset-dir", "../datasets",
    "Path to where the data lives.")
_FILE_PATH = flags.DEFINE_string(
    "file-path", "../datasets/train_dataset.npy", "Name of dataset file to load.")
_PRETRAIN_PATH = flags.DEFINE_string(
    "pretrain-path", "train_dataset.npy", "Name of dataset file to load.")
_GUESS_PREFIX = flags.DEFINE_string(
    "guess-prefix", "guess", "Name of dataset file to load.")
_BASEMODEL_PATH = flags.DEFINE_string(
    "basemodel-path", "guess", "Name of dataset file to load.")
_NUM_TRIALS = flags.DEFINE_integer(
    'num-trials', 100, 'Number of generations per prompt.')
_TOKEN_NUM = flags.DEFINE_integer(
    'token-num', 10, 'Number of soft prompt tokens.')
_PREFIX_LEN = flags.DEFINE_integer(
    'prefix-len', 50, 'Number of soft prompt tokens.')
_SUFFIX_LEN = flags.DEFINE_integer(
    'suffix-len', 50, 'Number of soft prompt tokens.')
_CHUNK = flags.DEFINE_integer(
    'chunk', 1000, 'Number of used prefixes')
_BATCH_SIZE = flags.DEFINE_integer(
    'bs', 64, 'Number of used prefixes')
_SEED = flags.DEFINE_integer(
    'seed', 2022, 'Number of used prefixes')

# _SUFFIX_LEN = 50
# _PREFIX_LEN = 50
# print(flags.FLAGS['token-num'])
print('loading model...')
# _MODEL = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
value = 100
model = SoftPromptModel(token_num=value, basemodel_path='/home/zhangzhexin/huggingface_pretrained_models/gpt-neo-1.3B')
ckpt = torch.load('./save/token100_maxlossToken5_alpha0.7_needlosslen50/seed42_realbs32_1gpu_lr1e-3_warmup500_lineardecay_maxepoch20/_epoch12_valloss0.74990_extraloss0.60754.bin', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
# _MODEL = transformers.AutoModelForCausalLM.from_pretrained("/data/zhangzhexin/huggingface_pretrained_models/gpt-neo-1.3B")
_MODEL = model.model
_MODEL = _MODEL.cuda().eval()
# _MODEL.resize_token_embeddings(50257 + value)
raw_embedding = _MODEL.get_input_embeddings()
vocab_size = raw_embedding.weight.size(0)
print(f'model vocab size:{vocab_size}')
# raw_embedding.weight[-value:] = model.soft_embeds.detach().clone()
raw_embedding.weight.data = torch.cat([raw_embedding.weight.data, model.soft_embeds.cuda()], dim=0)
_MODEL.half()
print('load model successfully')

def generate_for_prompts(
    prompts: np.ndarray, batch_size: int=64) -> Tuple[np.ndarray, np.ndarray]:
    """Generates suffixes given `prompts` and scores using their likelihood.

    Args:
    prompts: A np.ndarray of shape [num_prompts, prefix_length]. These
        provide the context for generating each suffix. Each value should be an
        int representing the token_id. These are directly provided by loading the
        saved datasets from extract_dataset.py.
    batch_size: The number of prefixes to generate suffixes for
        sequentially.

    Returns:
        A tuple of generations and their corresponding likelihoods.
        The generations take shape [num_prompts, _SUFFIX_LEN].
        The likelihoods take shape [num_prompts]
    """
    generations = []
    losses = []
    generation_len = _SUFFIX_LEN.value + _PREFIX_LEN.value
    prompt_ids = torch.arange(vocab_size, vocab_size + value, dtype=torch.long)

    bad_words_ids = [[i] for i in range(vocab_size, vocab_size + value)]
    # bad_words_ids = None
    if bad_words_ids is not None:
        vocab_extra_len = len(bad_words_ids)
    else:
        vocab_extra_len = 0
        
    generation_len += len(prompt_ids)
    all_token_losses = []
    for i, off in enumerate(range(0, len(prompts), batch_size)):
        prompt_batch = prompts[off:off+batch_size]
        logging.info(
            "Generating for batch ID {:05} of size {:04}".format(i, len(prompt_batch)))
        prompt_batch = np.stack(prompt_batch, axis=0)
        prompt_batch = prompt_batch[:, -_PREFIX_LEN.value:]
        # logging.info(f'prefix shape:{prompt_batch.shape}')
        input_ids = torch.tensor(prompt_batch, dtype=torch.int64)
        repeat_prompt_ids = prompt_ids.unsqueeze(0).repeat(input_ids.size(0), 1)
        input_ids = torch.cat([repeat_prompt_ids, input_ids], dim=1)
        with torch.no_grad():
            # 1. Generate outputs from the model
            gen_outputs = _MODEL.generate(
                input_ids.cuda(),
                max_length=generation_len,
                min_length=generation_len,
                num_beams=1,
                do_sample=True, 
                top_k=0,
                top_p=0.7,
                temperature=0.8,
                bad_words_ids=bad_words_ids,
                repetition_penalty=1.0,
                length_penalty=0,
                return_dict_in_generate=True,
                # output_scores=True,
                pad_token_id=50256  # Silences warning.
            )
            generated_tokens = gen_outputs.sequences.cpu().detach()

            # print(gen_outputs.scores)
            # 2. Compute each sequence's probability, excluding EOS and SOS.
            labels = generated_tokens.cuda().clone()
            labels[:, :len(prompt_ids)] = -100
            outputs = _MODEL(
                generated_tokens.cuda(),
                labels=labels,
            )
            logits = outputs.logits.cpu().detach()
            # logging.info(f'logits size:{logits.size()}')
            # print(f'logits size:{logits.size()}')
            # logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float()
            # 注意尽管有bad_words_ids，返回的logits中并没有对这些位置设置为负无穷，因此会影响概率的计算
            logits = logits[:, -_SUFFIX_LEN.value-1:-1].reshape((-1, logits.shape[-1])).float()

            # 不考虑extra_len的logits
            # print(logits[:, -extra_len:].mean())
            if vocab_extra_len > 0:
                logits = logits[:, :-vocab_extra_len]
            # print(generated_tokens)
            # loss_per_token = torch.nn.functional.cross_entropy(
            #     logits, generated_tokens[:, 1:].flatten(), reduction='none')
            loss_per_token = torch.nn.functional.cross_entropy(
                logits, generated_tokens[:, -_SUFFIX_LEN.value:].flatten(), reduction='none')
            # loss_per_token = loss_per_token.reshape((-1, generation_len - 1))[:,-_SUFFIX_LEN-1:-1]
            # loss_per_token = loss_per_token.reshape((-1, generation_len - 1))[:,-_SUFFIX_LEN.value:]
            loss_per_token = loss_per_token.reshape((-1, _SUFFIX_LEN.value))

            likelihood = loss_per_token.mean(1)
            
            generations.extend(generated_tokens.numpy())
            losses.extend(likelihood.numpy())
            all_token_losses.extend(loss_per_token.numpy())
            
    return np.atleast_2d(generations), np.atleast_2d(losses).reshape((len(generations), -1)), np.atleast_2d(all_token_losses)


def write_array(
    file_path: str, array: np.ndarray, unique_id: Union[int, str]):
    """Writes a batch of `generations` and `losses` to a file.

    Formats a `file_path` (e.g., "/tmp/run1/batch_{}.npy") using the `unique_id`
    so that each batch goes to a separate file. This function can be used in
    multiprocessing to speed this up.

    Args:
        file_path: A path that can be formatted with `unique_id`
        array: A numpy array to save.
        unique_id: A str or int to be formatted into `file_path`. If `file_path`
          and `unique_id` are the same, the files will collide and the contents
          will be overwritten.
    """
    file_ = file_path.format(unique_id)
    np.save(file_, array)


def load_prompts(file_path: str) -> np.ndarray:
    """Loads prompts from the file pointed to `dir_` and `file_name`."""
    return np.load(file_path).astype(np.int64)[:, -_PREFIX_LEN.value:]


def main(_):
    seed_everything(_SEED.value)
    experiment_base = os.path.join(_ROOT_DIR.value, _EXPERIMENT_NAME.value)
    generations_base = os.path.join(experiment_base, "generations")
    os.makedirs(generations_base, exist_ok=True)
    losses_base = os.path.join(experiment_base, "losses")
    os.makedirs(losses_base, exist_ok=True)
    all_token_losses_base = os.path.join(experiment_base, "all_token_losses")
    os.makedirs(all_token_losses_base, exist_ok=True)
    
    if _CHUNK.value != 0:
        prompts = load_prompts(_FILE_PATH.value)[-_CHUNK.value:]
        # if _CHUNK.value > 0:
        #     prompts = load_prompts(_DATASET_DIR.value, "train_prefix.npy")[:_CHUNK.value]
        # else:
        #     prompts = load_prompts(_DATASET_DIR.value, "train_prefix.npy")[_CHUNK.value:]
        # prompts = load_prompts(_DATASET_DIR.value, "train_prefix.npy")[-1000:-999]
    else:
        prompts = load_prompts(_FILE_PATH.value)

    # We by default do not overwrite previous results.
    all_generations, all_losses = [], []
    # if not all([os.listdir(generations_base), os.listdir(losses_base)]):
    for trial in tqdm(range(_NUM_TRIALS.value)):
        os.makedirs(experiment_base, exist_ok=True)
        generations, losses, all_token_losses = generate_for_prompts(prompts, batch_size=_BATCH_SIZE.value)

        generation_string = os.path.join(generations_base, "{}.npy")
        losses_string = os.path.join(losses_base, "{}.npy")
        all_token_losses_string = os.path.join(all_token_losses_base, "{}.npy")

        write_array(generation_string, generations, trial)
        write_array(losses_string, losses, trial)
        write_array(all_token_losses_string, all_token_losses, trial)

        all_generations.append(generations)
        all_losses.append(losses)
    generations = np.stack(all_generations, axis=1)
    losses = np.stack(all_losses, axis=1)
    
if __name__ == "__main__":
    app.run(main)
