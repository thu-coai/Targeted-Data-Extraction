import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import transformers

class SoftPromptModel(torch.nn.Module):
    def __init__(self, args=None, token_num=None, basemodel_path=None) -> None:
        super().__init__()
        
        # model = transformers.AutoModelForCausalLM.from_pretrained("/data/zhangzhexin/huggingface_pretrained_models/gpt-neo-1.3B")
        # model = transformers.AutoModelForCausalLM.from_pretrained("/home/zhangzhexin/huggingface_pretrained_models/gpt-neo-1.3B")
        if basemodel_path is None:
            model = transformers.AutoModelForCausalLM.from_pretrained(args.basemodel_path)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(basemodel_path)

        # model.config.pad_token_id = 50256
        self.model = model
        self.args = args
        raw_embedding = model.get_input_embeddings()
        if args is None:
            t = token_num
        else:
            t = args.token_num
            
        self.soft_embeds = torch.nn.Parameter(raw_embedding.weight[:t].clone().detach())

        for n, p in self.model.named_parameters():
            p.requires_grad_(False)
        
        
    def forward(self, input_ids=None):
        emb_layer = self.model.get_input_embeddings()
        inputs_embeds = emb_layer(input_ids)
        soft_embeds = self.soft_embeds.unsqueeze(0).repeat(input_ids.size(0), 1, 1)
        inputs_embeds = torch.cat([soft_embeds, inputs_embeds], dim=1)
        labels = input_ids.detach().clone()
        need_loss_len = self.args.need_loss_len
        labels[:, :-need_loss_len] = -100
        bs = input_ids.size(0)
        soft_token_num = soft_embeds.size(1)
        cat_labels = torch.full((bs, soft_token_num), -100, dtype=torch.long, device=input_ids.device)
        labels = torch.cat([cat_labels, labels], dim=1)
        # print(input_ids.size(), inputs_embeds.size(), labels.size(), labels)
        outputs = self.model(inputs_embeds=inputs_embeds, labels=labels)
        if not self.args.useneg:
            if self.args.maxloss_tokennum:
                # 找出Loss最高的几个token额外计算Loss，相当于在这些token上分配更高的权重
                logits = outputs.logits
                fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                shift_labels = labels[:, 1:]
                shift_logits = logits[:, :-1]
                losses = fct(shift_logits.transpose(1, 2), shift_labels)[:,-need_loss_len:]
                maxtopk_losses = torch.topk(losses, self.args.maxloss_tokennum, dim=-1)[0]
                total_loss = outputs.loss + self.args.extraloss_alpha * maxtopk_losses.mean()
                # print(outputs.loss.item(), maxtopk_losses.mean().item())
                outputs.loss = total_loss
                outputs.__setitem__('maxloss', self.args.extraloss_alpha * maxtopk_losses.mean())
                # outputs.maxloss = self.args.extraloss_alpha * maxtopk_losses.mean()
                # print(f'outputs keys:{outputs.keys()}')
                # print(f'outputs.maxloss:{outputs.maxloss}')
                return outputs
            elif self.args.minloss:
                logits = outputs.logits
                fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                shift_labels = labels[:, 1:]
                shift_logits = logits[:, :-1]
                losses = fct(shift_logits.transpose(1, 2), shift_labels)[:, -need_loss_len:]
                mask = losses > self.args.minloss
                select_loss = torch.masked_select(losses, mask)
                if select_loss.numel() > 0:
                    total_loss = outputs.loss + self.args.extraloss_alpha * select_loss.mean()
                    # print(outputs.loss.item(), select_loss.mean().item())
                    outputs.loss = total_loss
                    outputs.minloss = self.args.extraloss_alpha * select_loss.mean()
                    return outputs
                else:
                    outputs.minloss = torch.tensor(0, dtype=torch.float, device=outputs.loss.device)
                    # print(outputs.loss.item(), outputs.minloss.item())
                    return outputs
            else:
                return outputs

        else:
            logits = outputs.logits
            fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            shift_labels = labels[:, 1:]
            shift_logits = logits[:, :-1]
            losses = fct(shift_logits.transpose(1, 2), shift_labels)
            batch_token_num = (labels != -100).sum(-1)
            # print(losses, batch_token_num)
            losses = losses.sum(-1) / batch_token_num
            outputs.each_loss = losses
            return outputs
        
        
class PrefixModel(torch.nn.Module):
    def __init__(self, args=None, token_num=None) -> None:
        super().__init__()
        
        # model = transformers.AutoModelForCausalLM.from_pretrained("/data/zhangzhexin/huggingface_pretrained_models/gpt-neo-1.3B")
        model = transformers.AutoModelForCausalLM.from_pretrained("/home/zhangzhexin/huggingface_pretrained_models/gpt-neo-1.3B")
        config = model.config
        # print(config)
        self.num_token = args.token_num
        self.n_layer = config.num_layers
        self.n_embd = config.hidden_size
        self.mid_dim = 2048
        # self.prefix_dropout = args.prefix_dropout
        self.prefix_dropout = 0.0
        self.match_n_head = config.num_heads
        self.match_n_embd = self.n_embd // self.match_n_head

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.input_tokens = nn.Parameter(torch.arange(
            self.num_token).long(), requires_grad=False)

        self.wte = nn.Embedding(self.num_token, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_layer * 2 * self.n_embd))
        # model.config.pad_token_id = 50256
        self.model = model
        self.args = args
        # raw_embedding = model.get_input_embeddings()
        # if args is None:
        #     t = token_num
        # else:
        #     t = args.token_num
            
        # self.soft_embeds = torch.nn.Parameter(raw_embedding.weight[:t].clone().detach())

        for n, p in self.model.named_parameters():
            p.requires_grad_(False)
            
    def get_past_key_values(self, batch_size):
        pvs = []
        input_tokens = self.input_tokens.unsqueeze(0).repeat(batch_size, 1)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz=1, seqlen, layer*emb*2
        _, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(batch_size, seqlen, self.n_layer * 2, self.match_n_head,
                                            self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(self.n_layer)
        # print(len(past_key_values), past_key_values[0].size())
        pvs.extend(past_key_values)

        return pvs
        
        
    def forward(self, input_ids=None):
        past_key_values = self.get_past_key_values(batch_size=input_ids.size(0))
        labels = input_ids.detach().clone()
        labels[:, :-50] = -100
        outputs = self.model(input_ids=input_ids, labels=labels, past_key_values=past_key_values)
        return outputs

        emb_layer = self.model.get_input_embeddings()
        inputs_embeds = emb_layer(input_ids)
        soft_embeds = self.soft_embeds.unsqueeze(0).repeat(input_ids.size(0), 1, 1)
        inputs_embeds = torch.cat([soft_embeds, inputs_embeds], dim=1)
        labels = input_ids.detach().clone()
        labels[:, :-50] = -100
        bs = input_ids.size(0)
        soft_token_num = soft_embeds.size(1)
        cat_labels = torch.full((bs, soft_token_num), -100, dtype=torch.long, device=input_ids.device)
        labels = torch.cat([cat_labels, labels], dim=1)
        # print(input_ids.size(), inputs_embeds.size(), labels.size(), labels)
        outputs = self.model(inputs_embeds=inputs_embeds, labels=labels)
        if not self.args.useneg:
            if self.args.maxloss_tokennum:
                # 找出Loss最高的几个token额外计算Loss，相当于在这些token上分配更高的权重
                logits = outputs.logits
                fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                shift_labels = labels[:, 1:]
                shift_logits = logits[:, :-1]
                losses = fct(shift_logits.transpose(1, 2), shift_labels)[:, -50:]
                maxtopk_losses = torch.topk(losses, self.args.maxloss_tokennum, dim=-1)[0]
                total_loss = outputs.loss + self.args.extraloss_alpha * maxtopk_losses.mean()
                # print(outputs.loss.item(), maxtopk_losses.mean().item())
                outputs.loss = total_loss
                outputs.maxloss = maxtopk_losses.mean()
                return outputs
            elif self.args.minloss:
                logits = outputs.logits
                fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                shift_labels = labels[:, 1:]
                shift_logits = logits[:, :-1]
                losses = fct(shift_logits.transpose(1, 2), shift_labels)[:, -50:]
                mask = losses > self.args.minloss
                select_loss = torch.masked_select(losses, mask)
                if select_loss.numel() > 0:
                    total_loss = outputs.loss + self.args.extraloss_alpha * select_loss.mean()
                    # print(outputs.loss.item(), select_loss.mean().item())
                    outputs.loss = total_loss
                    outputs.minloss = select_loss.mean()
                    return outputs
                else:
                    outputs.minloss = torch.tensor(0, dtype=torch.float, device=outputs.loss.device)
                    # print(outputs.loss.item(), outputs.minloss.item())
                    return outputs
            else:
                return outputs

        else:
            logits = outputs.logits
            fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            shift_labels = labels[:, 1:]
            shift_logits = logits[:, :-1]
            losses = fct(shift_logits.transpose(1, 2), shift_labels)
            batch_token_num = (labels != -100).sum(-1)
            # print(losses, batch_token_num)
            losses = losses.sum(-1) / batch_token_num
            outputs.each_loss = losses
            return outputs
  