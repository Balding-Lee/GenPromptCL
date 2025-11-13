import math
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
# from transformers import BertConfig, BertForMaskedLM
from transformers import RobertaConfig, RobertaForMaskedLM


class MaksedLanguageModel(nn.Module):

    def __init__(self, model_path, tokenizer, verbalizer_ids):
        super().__init__()
        self.model_config = RobertaConfig.from_pretrained(model_path)
        self.model_config.output_hidden_states = True
        self.model = RobertaForMaskedLM.from_pretrained(model_path, config=self.model_config)
        self.tokenizer = tokenizer
        self.verbalizer_ids = verbalizer_ids

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        # shape: (batch_size, max_length, vocab_size)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        last_hidden_states = outputs.hidden_states[-1]

        mask_token_index = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)

        masked_logits = logits[mask_token_index[0], mask_token_index[1], :]
        masked_hidden_states = last_hidden_states[mask_token_index[0], mask_token_index[1], :]

        verbalizer_logits = masked_logits[:, self.verbalizer_ids]

        return verbalizer_logits, masked_hidden_states, last_hidden_states[:, 0, :]


class DomainClassifier(nn.Module):
    def __init__(self, lr, in_size=768):
        super().__init__()
        self.dense = nn.Linear(in_size, in_size)
        self.dropout = nn.Dropout(0.0)
        self.out_proj = nn.Linear(in_size, 1)  # 2 domains
        self.optimizer = AdamW(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DomainDiscriminators(nn.Module):
    def __init__(self, num_domains, lr, in_size=768):
        super().__init__()
        self.domain_num = num_domains - 1
        self.domain_class = nn.ModuleList([DomainClassifier(lr, in_size) for _ in
                                           range((self.domain_num - 1) * self.domain_num // 2)])

    def forward(self, features, global_step=None, total_step=None):
        if global_step is not None and total_step is not None:
            progress = float(global_step) / float(total_step)
            lmda = 2 / (1 + math.exp(-5 * progress)) - 1
        else:
            lmda = 1.

        j_idx = 0
        loss_domain_disc_list_ = []
        error_domain_disc_list_ = []
        for i in range(self.domain_num):
            domain_t = torch.ones(features[i].size(0), requires_grad=False).type(torch.FloatTensor).to(
                features[0].device)
            for j in range(self.domain_num):
                if i < j:
                    domain_f = torch.zeros(features[j].size(0), requires_grad=False).type(torch.FloatTensor).to(
                        features[0].device)
                    logits_t = self.domain_class[j_idx](features[i].detach()).squeeze(1)
                    logits_f = self.domain_class[j_idx](features[j].detach()).squeeze(1)
                    error_domain_dis = ((1 - F.sigmoid(logits_t)).mean() + F.sigmoid(logits_f).mean()) * 0.5

                    domain_discriminator_loss = ((
                        F.binary_cross_entropy_with_logits(logits_t, domain_t) +
                        F.binary_cross_entropy_with_logits(logits_f, domain_f)) * 0.5
                    )

                    self.domain_class[j_idx].optimizer.zero_grad()
                    domain_discriminator_loss.backward()
                    self.domain_class[j_idx].optimizer.step()
                    j_idx += 1
                    error_domain_disc_list_.append(error_domain_dis.detach().item())
                    loss_domain_disc_list_.append(domain_discriminator_loss.detach().item())
        domdis_losses = []
        j_idx = 0
        for i in range(self.domain_num):
            for j in range(self.domain_num):
                if i < j:
                    domain_t = torch.ones(features[j].size(0), requires_grad=False).type(torch.FloatTensor).to(
                        features[0].device)
                    domain_f = torch.zeros(features[i].size(0), requires_grad=False).type(torch.FloatTensor).to(
                        features[0].device)

                    logits_t = self.domain_class[j_idx](features[i]).detach().squeeze(1)
                    logits_f = self.domain_class[j_idx](features[j]).detach().squeeze(1)

                    domain_discriminator_loss = ((
                        F.binary_cross_entropy_with_logits(logits_t, domain_f) +
                        F.binary_cross_entropy_with_logits(logits_f, domain_t))
                            * 0.5
                    )

                    domdis_losses.append(domain_discriminator_loss * lmda)
                    j_idx += 1

        return torch.stack(domdis_losses).mean(), error_domain_disc_list_, loss_domain_disc_list_


class DomainKL(nn.Module):
    def __init__(self, num_domains):
        super().__init__()
        self.domain_num = num_domains - 1

    def forward(self, features):
        kl_div_list = []
        for i in range(self.domain_num):
            for j in range(self.domain_num):
                if i != j:
                    p = F.softmax(features[i], dim=-1)
                    q = F.softmax(features[j], dim=-1)
                    kl_div = F.kl_div(torch.mean(p, dim=0).log(), torch.mean(q, dim=0))
                    kl_div_list.append(kl_div)

        return torch.stack(kl_div_list).mean()
