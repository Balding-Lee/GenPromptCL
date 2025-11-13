'''
@Time : 2024/3/20 14:37
@Auth : Qizhi Li
'''
import os
import math
import torch
import random
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
import ignite.distributed as idist
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification, RobertaModel


class CoVWeightingLoss(nn.Module):
    """
        Wrapper of the BaseLoss which weighs the losses to the Cov-Weighting method,
        where the statistics are maintained through Welford's algorithm. But now for 32 losses.
    """

    def __init__(self, mean_sort, mean_decay_param, device, save_losses=False, target_domain=None):
        super().__init__()
        self.device = device
        self.save_losses = save_losses
        self.target_domain = target_domain
        self.num_losses = 3

        # How to compute the mean statistics: Full mean or decaying mean.
        self.mean_decay = True if mean_sort == 'decay' else False
        self.mean_decay_param = mean_decay_param

        self.current_iter = -1
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)

        # Initialize all running statistics at 0.
        self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
            self.device)
        self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
            self.device)
        self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_std_l = None

    def forward(self, unweighted_losses, losses_names=None, iteration=None):
        # Retrieve the unweighted losses.
        # unweighted_losses = super(CoVWeightingLoss, self).forward(pred, target)

        # Put the losses in a list. Just for computing the weights.
        L = torch.tensor(unweighted_losses, requires_grad=False).to(self.device)

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        if not self.train:
            return torch.sum(L)

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = (torch.ones((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
                           / self.num_losses)
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay:
            mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]

        if self.save_losses:
            assert iteration is not None
            assert len(weighted_losses) == len(losses_names)

            losses_curve_path = '../results/losses_curve_{}.txt'.format(self.target_domain)
            weight_curve_path = '../results/weight_curve_{}.txt'.format(self.target_domain)

            with open(losses_curve_path, 'a') as f:
                f.write('Iter: %d\t' % iteration)
                for i in range(len(weighted_losses)):
                    f.write('%s: %.4f\t' % (losses_names[i], weighted_losses[i]))
                f.write('\n')

            with open(weight_curve_path, 'a') as f:
                f.write('Iter: %d\t' % iteration)
                for i in range(len(unweighted_losses)):
                    f.write('%s: %f\t' % (losses_names[i], self.alphas[i]))

                f.write('\n')

        loss = sum(weighted_losses)
        return loss


class SentenceOrthogonality(nn.Module):
    def __init__(self, in_size, out_size, device, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.hidden_size = 768

        self.encode_linear = nn.Linear(in_size, out_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.mse_loss = nn.MSELoss()
        if num_classes == 2:
            self.ce_loss = nn.BCELoss()
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.1)

    def forward(self, sentence_representations, labels):
        outputs = self.sigmoid(self.encode_linear(sentence_representations))

        sentence_similarity = self.sigmoid(torch.matmul(outputs, outputs.T))

        num_classes = self.num_classes
        # num_classes = len(torch.unique(labels))

        # shape: (batch_size * num_domains, num_classes)
        one_hot = torch.eye(num_classes)[labels.cpu()]
        one_hot = one_hot.to(self.device)

        # shape: (batch_size * num_domains, batch_size * num_domains)
        label_similarity = torch.matmul(one_hot, one_hot.T)

        # label_similarity = torch.LongTensor(label_similarity).to(self.device)
        contrastive_loss = self.ce_loss(sentence_similarity, label_similarity)

        return contrastive_loss


class Model(nn.Module):

    def __init__(self, model_path, tokenizer, s_verbalizer_ids, d_verbalizer_ids):
        super().__init__()
        # self.model_config = BertConfig.from_pretrained(model_path)
        self.model_config = RobertaConfig.from_pretrained(model_path)
        self.model_config.output_hidden_states = True
        # self.model = BertForMaskedLM.from_pretrained(model_path, config=self.model_config)
        self.model = RobertaForMaskedLM.from_pretrained(model_path, config=self.model_config)
        self.tokenizer = tokenizer
        self.s_verbalizer_ids = s_verbalizer_ids
        self.d_verbalizer_ids = d_verbalizer_ids

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: (batch_size, max_length)
        :param attention_mask: (batch_size, max_length)
        """
        # shape: (batch_size, max_length, vocab_size)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        last_hidden_states = outputs.hidden_states[-1]

        # mask_token_index[0]: the i-th data
        # mask_token_index[1]: the index of [MASK] in the i-th data
        mask_token_index = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)

        # find the logits of [MASK]
        # shape: (batch_size * 2, vocab_size)
        # odd number: domain mask
        # even number: sentiment mask
        masked_logits = logits[mask_token_index[0], mask_token_index[1], :]

        # shape: (batch_size, vocab_size)
        # logits of domain mask
        domain_masked_logits = masked_logits[::2]
        # logits of sentiment mask
        sentiment_masked_logits = masked_logits[1::2]

        # shape: (batch_size * 2, hidden_size)
        masked_hidden_states = last_hidden_states[mask_token_index[0], mask_token_index[1], :]
        sentiment_masked_hidden_states = masked_hidden_states[1::2]
        # shape: (batch_size, hidden_size)
        cls_hidden_states = last_hidden_states[:, 0, :]

        # Extract the logits of the words in the verbalizer at the [MASK] position
        # shape: (batch_size, verbalizer_size)
        s_verbalizer_logits = sentiment_masked_logits[:, self.s_verbalizer_ids]
        d_verbalizer_logits = domain_masked_logits[:, self.d_verbalizer_ids]

        return s_verbalizer_logits, d_verbalizer_logits, sentiment_masked_hidden_states, cls_hidden_states, last_hidden_states


class BaselineMlmModel(nn.Module):

    def __init__(self, model_path, tokenizer, s_verbalizer_ids):
        super().__init__()
        # self.model_config = BertConfig.from_pretrained(model_path)
        self.model_config = RobertaConfig.from_pretrained(model_path)
        self.model_config.output_hidden_states = True
        # self.model = BertForMaskedLM.from_pretrained(model_path, config=self.model_config)
        self.model = RobertaForMaskedLM.from_pretrained(model_path, config=self.model_config)
        self.tokenizer = tokenizer
        self.s_verbalizer_ids = s_verbalizer_ids

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: (batch_size, max_length)
        :param attention_mask: (batch_size, max_length)
        """
        # shape: (batch_size, max_length, vocab_size)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # mask_token_index[0]: the i-th data
        # mask_token_index[1]: the index of [MASK] in the i-th data
        mask_token_index = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)

        # find the logits of [MASK]
        # shape: (batch_size * 2, vocab_size)
        # odd number: domain mask
        # even number: sentiment mask
        masked_logits = logits[mask_token_index[0], mask_token_index[1], :]

        # Extract the logits of the words in the verbalizer at the [MASK] position
        # shape: (batch_size, verbalizer_size)
        s_verbalizer_logits = masked_logits[:, self.s_verbalizer_ids]

        return s_verbalizer_logits


class BaselineClsModel(nn.Module):
    def __init__(self, model_path, num_outputs=2):
        super().__init__()
        self.model_config = RobertaConfig.from_pretrained(model_path, num_labels=num_outputs)
        self.model_config.output_hidden_states = True
        self.model = RobertaForSequenceClassification.from_pretrained(model_path, config=self.model_config)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # shape: (batch * 2, num_outputs)
        logits = outputs.logits

        return logits


class DeepCORAL(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super().__init__()
        self.backbone = RoBERTaWithCLS(model_path)
        self.fc = nn.Linear(256, num_classes)

        # initialize according to CORAL paper experiment
        self.fc.weight.data.normal_(0, 0.005)

    def forward(self, source, source_att, target, target_att):
        source = self.backbone(source, source_att)
        source = self.fc(source)

        target = self.backbone(target, target_att)
        target = self.fc(target)
        return source, target


class RoBERTaWithCLS(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model_config = RobertaConfig.from_pretrained(model_path)
        self.model = RobertaModel.from_pretrained(model_path, config=self.model_config)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_ids, att_masks):
        x = self.model(input_ids, attention_mask=att_masks).pooler_output
        x = self.classifier(x)
        return x


def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)

    return loss


@torch.no_grad()
def distributed_sinkhorn(out, epsilon=0.05, num_iterations=3, normalization='col'):
    # https://github.com/facebookresearch/swav/blob/main/main_swav.py

    Q = torch.exp(out / epsilon)  # Q is B-by-K (B = batch size, K = queue size)
    B = Q.shape[0] * idist.get_world_size()
    K = Q.shape[1]

    # make the matrix sums to 1
    Q /= idist.all_reduce(torch.sum(Q))

    if normalization == 'col':
        for it in range(num_iterations):
            # normalize each row: total weight per prototype must be 1/K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

            # normalize each column: total weight per sample must be 1/B
            Q /= idist.all_reduce(torch.sum(Q, dim=0, keepdim=True))
            Q /= K

        Q *= K  # the colomns must sum to 1 so that Q is an assignment
    else:
        for it in range(num_iterations):
            # normalize each column: total weight per sample must be 1/B
            Q /= idist.all_reduce(torch.sum(Q, dim=0, keepdim=True))
            Q /= K

            # normalize each row: total weight per prototype must be 1/K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q


def get_mlp(*layers, last_bn=False):
    modules = []
    for i in range(len(layers) - 1):
        modules.append(nn.Linear(layers[i], layers[i + 1], bias=False))
        if i < len(layers) - 2:
            modules.append(nn.BatchNorm1d(layers[i + 1]))
            modules.append(nn.ReLU(inplace=True))
        elif last_bn:
            modules.append(nn.BatchNorm1d(layers[i + 1], affine=False))
    return nn.Sequential(*modules)


class RoBERTa(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model_config = RobertaConfig.from_pretrained(model_path)
        self.model = RobertaModel.from_pretrained(model_path, config=self.model_config)

    def forward(self, input_ids, att_masks):
        x = self.model(input_ids, attention_mask=att_masks).pooler_output
        return x


class SwAV(nn.Module):
    def __init__(self,
                 model_path,
                 temperature: float = 0.2,
                 n_prototypes: int = 512,
                 sinkhorn_iter: int = 3,
                 ):
        super().__init__()
        self.backbone = RoBERTa(model_path)
        self.prototypes = nn.Linear(256, n_prototypes, bias=False)
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.projector = get_mlp(768, 512, 256, last_bn=True)

    def forward(self, input_ids, att_masks, synonym_input_ids, synonym_att_masks):
        with torch.no_grad():
            w = F.normalize(self.prototypes.weight.data.clone())
            self.prototypes.weight.copy_(w)
        z1 = F.normalize(self.projector(self.backbone(input_ids, att_masks)))
        z2 = F.normalize(self.projector(self.backbone(synonym_input_ids, synonym_att_masks)))
        scores1 = self.prototypes(z1)
        scores2 = self.prototypes(z2)
        q1 = distributed_sinkhorn(scores1.detach(), normalization='row', num_iterations=self.sinkhorn_iter)
        q2 = distributed_sinkhorn(scores2.detach(), normalization='row', num_iterations=self.sinkhorn_iter)
        logp1 = F.log_softmax(scores1.div(self.temperature), dim=1)
        logp2 = F.log_softmax(scores2.div(self.temperature), dim=1)
        loss = -(torch.sum(q1.mul(logp2), dim=1) + torch.sum(q2.mul(logp1), dim=1)).mean()
        return loss


class Classifier(nn.Module):
    def __init__(self, hidden_size=768, output_size=2):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, logits):
        return self.classifier(logits)
