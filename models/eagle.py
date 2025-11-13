'''
@Time : 2024/4/2 15:01
@Auth : Qizhi Li
'''
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaForSequenceClassification


class Backbone(nn.Module):
    def __init__(self, model_path, input_size=768, output_size=300, num_outputs=2):
        super().__init__()
        self.model_config = RobertaConfig.from_pretrained(model_path, num_labels=num_outputs)
        self.model_config.output_hidden_states = True
        self.model = RobertaForSequenceClassification.from_pretrained(model_path, config=self.model_config)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # shape: (batch * 2, num_outputs)
        logits = outputs.logits
        # shape: (batch_size, hidden_size)
        cls_representations = outputs.hidden_states[-1][:, 0, :]

        # shape: (batch_size, output_size)
        z = self.linear(cls_representations)

        return logits, z, cls_representations


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReverse(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReverse, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)


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
        # 因为有个目标域所以要-1
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
            # 论文中提到的索引小的标签为0
            domain_t = torch.ones(features[i].size(0), requires_grad=False).type(torch.FloatTensor).to(
                features[0].device)
            for j in range(self.domain_num):
                # 论文中提到的索引大的标签为1
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


class SimCLR(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, zi, zj):
        batch_size = zi.size(0)

        # 计算相似性
        representations = torch.cat([zi, zj], dim=0)
        similarity_matrix = torch.matmul(representations, representations.t())
        # 对角线部分除以温度参数
        similarity_matrix = similarity_matrix / self.temperature

        # 构建目标对
        mask = torch.eye(batch_size * 2, dtype=torch.bool, device=zi.device)
        neg_mask = ~mask

        # 计算对比损失
        logits_aa = similarity_matrix.masked_select(mask).view(batch_size, -1)
        logits_bb = similarity_matrix.masked_select(mask).view(batch_size, -1)
        logits_ab = similarity_matrix.masked_select(neg_mask).view(batch_size, -1)
        logits_ba = similarity_matrix.masked_select(neg_mask).view(batch_size, -1)

        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)

        logits = torch.cat([logits_a, logits_b], dim=0)

        targets = torch.arange(batch_size, device=zi.device)
        targets = torch.cat([targets, targets], dim=0)

        return nn.CrossEntropyLoss()(logits, targets)

