import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaConfig, RobertaModel


class SimCSEModel(nn.Module):
    """ SimCSE无监督模型定义 """

    def __init__(self, model_path, pooling, device):
        super(SimCSEModel, self).__init__()
        config = RobertaConfig.from_pretrained(model_path)
        self.bert = RobertaModel.from_pretrained(model_path, config=config)
        self.pooling = pooling
        self.device = device

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask, output_hidden_states=True)
        # shape: (batch_size, hidden_size)
        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]
        # shape: (batch_size, hidden_size)
        if self.pooling == 'pooler':
            return out.pooler_output
        # shape: (batch_size, hidden_size)
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
        # shape: (batch_size, hidden_size)
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)
            last = out.hidden_states[-1].transpose(1, 2)
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)

    def simcse_unsup_loss(self, y_pred):
        """  无监督的损失函数  """
        # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
        y_true = torch.arange(y_pred.shape[0]).to(self.device)  # [batch_size*2]
        y_true = (y_true - y_true % 2 * 2) + 1
        # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        sim = sim - torch.eye(y_pred.shape[0]).to(self.device) * 1e12
        # 相似度矩阵除以温度系数
        sim = sim / 0.05
        # 计算相似度矩阵与y_true的交叉熵损失
        loss = F.cross_entropy(sim, y_true)
        return loss


class Classifier(nn.Module):
    def __init__(self, hidden_size=768, output_size=2):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, logits):
        return self.classifier(logits)
