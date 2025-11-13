'''
@Time : 2024/4/28 19:03
@Auth : Qizhi Li
'''
import json
import os
import sys
import tqdm
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import autograd
from torch.optim import AdamW
import torch.nn.functional as F
from nltk.tokenize import word_tokenize

sys.path.append('..')
from models import GenPromptCL


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MLP(nn.Module):
    def __init__(self, embed, max_length, embedding_dim, hidden_dim, output_size=1):
        super().__init__()
        self.max_length = max_length
        self.embedding_dim = embedding_dim

        self.embedding_layer = nn.Embedding.from_pretrained(embed, freeze=True)
        lin1 = nn.Linear(max_length * embedding_dim, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        lin3 = nn.Linear(hidden_dim, output_size)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, input):
        # input shape: (num_samples, max_length)
        # shape: (num_samples, max_length, embedding_dim)
        input = self.embedding_layer(input)
        # shape: (num_samples, max_length * embedding_dim)
        out = input.view(input.shape[0], self.max_length * self.embedding_dim)
        # shape: (num_samples, 1)
        out = self._main(out)
        return out


def make_environment(texts, labels, word2id, max_length, device):
    input_ids = []
    for text in tqdm.tqdm(texts, desc='preparing environment'):
        ids = []
        words = word_tokenize(text)

        for word in words:
            ids.append(word2id[word])

        if len(words) > max_length:
            ids = ids[:max_length]
        else:
            ids.extend([word2id['<PAD>']] * (max_length - len(words)))

        input_ids.append(ids)

        return {
            'texts': torch.LongTensor(input_ids).to(device),
            'labels': torch.FloatTensor(labels).unsqueeze(-1).to(device)
        }


def make_nli_environment(sentence1, sentence2, labels, word2id,
                         single_sentence_max_length, device, target_domain):
    input_ids = []
    for i in tqdm.trange(len(sentence1), desc='preparing {} environment'.format(target_domain)):
        ids1, ids2 = [], []
        words1 = word_tokenize(sentence1[i])
        words2 = word_tokenize(sentence2[i])

        for word in words1:
            ids1.append(word2id[word])

        for word in words2:
            ids2.append(word2id[word])

        if len(words1) > single_sentence_max_length:
            ids1 = ids1[:single_sentence_max_length]
        else:
            ids1.extend([word2id['<PAD>']] * (single_sentence_max_length - len(words1)))

        if len(words2) > single_sentence_max_length:
            ids2 = ids2[:single_sentence_max_length]
        else:
            ids2.extend([word2id['<PAD>']] * (single_sentence_max_length - len(words2)))

        ids = ids1 + ids2
        input_ids.append(ids)

    return {
        'texts': torch.LongTensor(input_ids).to(device),
        'labels': torch.FloatTensor(labels).unsqueeze(-1).to(device)
    }


def pretty_print(*values):
    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


def read_imdb_csv(dir_path):
    data = pd.read_csv(dir_path, header=0)
    texts = data['sentence'].tolist()
    labels = data['labels'].tolist()

    return texts, labels


def load_imdb_dataset(dir_path, envs, word2id, max_length, device):
    test_texts, test_labels = read_imdb_csv(os.path.join(dir_path, 'test.csv'))

    envs.append(make_environment(test_texts, test_labels, word2id, max_length, device))

    return envs


def read_sst_tsv(dir_path):
    data = pd.read_csv(dir_path, sep='\t', header=0)
    texts = data['sentence'].tolist()
    labels = data['label'].tolist()

    return texts, labels


def load_sst_dataset(dir_path, envs, word2id, max_length, device):
    test_texts, test_labels = read_sst_tsv(os.path.join(dir_path, 'test.tsv'))

    envs.append(make_environment(test_texts, test_labels, word2id, max_length, device))
    return envs


def load_amazon_dataset(dir_path, target_domain, word2id, max_length, envs, device):
    train_texts, train_labels, train_domains = [], [], []
    test_texts, test_labels = [], []

    domains = os.listdir(dir_path)

    for domain in domains:
        if domain != target_domain:
            with open(os.path.join(dir_path, domain, 'all_data.txt'), 'r') as f:
                texts = f.readlines()

            train_domains.append(len(texts))
            for text in texts:
                line = text.strip().split(' ||| ')
                if len(line) == 2:
                    train_texts.append(line[0])
                    train_labels.append(int(line[1]))

    for num_samples in train_domains:
        # train_dataset = CustomDataset(train_texts[:num_samples], train_labels[:num_samples],
        #                               word2id, max_length)
        envs.append(make_environment(train_texts[:num_samples], train_labels[:num_samples], word2id, max_length, device))
        del train_texts[:num_samples]
        del train_labels[:num_samples]
        # train_datasets.append(train_dataset)

    if target_domain in domains:
        with open(os.path.join(dir_path, target_domain, 'test.txt'), 'r') as f:
            lines = f.readlines()

        for text in lines:
            line = text.strip().split(' ||| ')
            if len(line) == 2:
                test_texts.append(line[0])
                test_labels.append(int(line[1]))

        # test_dataset = CustomDataset(test_texts, test_labels, word2id, max_length)
        envs.append(make_environment(test_texts, test_labels, word2id, max_length, device))

    #     return train_datasets, test_dataset
    # else:
    #     return train_datasets

    return envs


def read_pheme_csv(dir_path):
    data = pd.read_csv(dir_path, header=0)
    texts = data['texts'].tolist()
    labels = data['labels'].tolist()
    return texts, labels


def load_rumour_dataset(dir_path, target_domain, word2id, max_length, envs, device):
    train_texts, train_labels, train_domains = [], [], []
    test_texts, test_labels = [], []

    domain_mapping = {
        'ch': 'charliehebdo',
        'f': 'ferguson',
        'gw': 'germanwings',
        'os': 'ottawashooting',
        's': 'sydneysiege',
    }

    domains = os.listdir(dir_path)

    for domain in domains:
        if domain != '{}.csv'.format(domain_mapping[target_domain]):
            texts, labels = read_pheme_csv(os.path.join(dir_path, domain))
            envs.append(make_environment(texts, labels, word2id, max_length, device))

    test_texts, test_labels = read_pheme_csv(os.path.join(dir_path, '{}.csv'.format(domain_mapping[target_domain])))
    envs.append(make_environment(test_texts, test_labels, word2id, max_length, device))

    return envs


def read_nli_csv(dir_path):
    df = pd.read_csv(dir_path, header=0)
    sentence1 = df['sentence1'].tolist()
    sentence2 = df['sentence2'].tolist()
    labels = df['label'].tolist()

    return sentence1, sentence2, labels


def load_mnli_dataset(dir_path, target_domain, word2id, single_sentence_max_length, envs, device):
    domains = os.listdir(dir_path)

    for domain in domains:
        if domain != target_domain:
            sentence1, sentence2, labels = read_nli_csv(os.path.join(dir_path, domain, 'train.csv'))
            envs.append(make_nli_environment(sentence1, sentence2, labels, word2id,
                                             single_sentence_max_length, device, domain))

    if target_domain in domains:
        test_s1, test_s2, test_labels = read_nli_csv(os.path.join(dir_path, target_domain, 'test.csv'))
        envs.append(make_nli_environment(test_s1, test_s2, test_labels, word2id,
                                         single_sentence_max_length, device, target_domain))

    return envs


def load_sick_and_nli_dataset(dir_path, word2id, single_sentence_max_length, envs, device):
    sentence1, sentence2, labels = read_nli_csv(dir_path)
    envs.append(make_nli_environment(sentence1, sentence2, labels, word2id,
                                     single_sentence_max_length, device, 'sick or snli'))

    return envs


def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


def nli_mean_nll(logits, y):
    return nn.functional.cross_entropy(logits, y)


def macro_f1(logits, y):
    preds = (logits > 0.).float()
    # Calculate F1 score for each class
    epsilon = 1e-7
    tp = (preds * y).sum(dim=0)
    fp = (preds * (1 - y)).sum(dim=0)
    fn = ((1 - preds) * y).sum(dim=0)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    # Calculate Macro-F1 score
    macro_f1_score = f1.mean()
    return macro_f1_score


def nli_macro_f1(logits, y):
    _, preds = torch.max(logits, dim=1)  # 获取每个样本的最大值索引

    # 如果 y 是 one-hot 编码的形式，需要将其转换为索引形式
    if y.dim() == 2 and y.size(1) > 1:
        y = torch.argmax(y, dim=1)

    # 计算 F1 分数
    epsilon = 1e-7
    num_classes = logits.size(1)
    f1_scores = []

    for c in range(num_classes):
        # 计算每个类别的 TP, FP 和 FN
        tp = ((preds == c) * (y == c)).sum().float()
        fp = ((preds == c) * (y != c)).sum().float()
        fn = (((preds != c)) * (y == c)).sum().float()

        # 计算 Precision 和 Recall
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        # 计算 F1 分数
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        f1_scores.append(f1)

    # 计算 Macro-F1 分数
    macro_f1_score = torch.mean(torch.stack(f1_scores))

    return macro_f1_score


def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()


def penalty(logits, y):
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def nli_penalty(logits, y, output_size):
    scale = torch.tensor(1.).to(device).requires_grad_()
    y = nn.functional.one_hot(y, num_classes=output_size).view(logits.shape).float()
    loss = nli_mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def evaluate(model, data_loader, tgt_domain):
    f1s = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc='test on {}'.format(tgt_domain)):
            input_ids = batch[0].to(device)
            att_masks = batch[1].to(device)
            # tgt = batch[4].tolist()
            tgt = batch[2].float().unsqueeze(-1).to(device)

            logits = model(input_ids, att_masks)
            f1 = macro_f1(logits, tgt)
            f1s.append(f1)

    model.train()
    macro_F1 = torch.stack(f1s).mean()
    return macro_F1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_domain',
                        choices=['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst',
                                 'ch', 'f', 'gw', 'os', 's'],
                        default='book')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_restarts', type=int, default=10)
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--steps', type=int, default=501)
    parser.add_argument('--seed', default=9)
    parser.add_argument('--cuda', default=0)
    args = parser.parse_args()

    seed = args.seed
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    target_domain = args.target_domain

    set_seed(seed)

    model_path = 'pretrained_parameters/roberta-base'
    amazon_dir_path = 'datasets/amazon'
    imdb_dir_path = 'datasets/imdb'
    sst_dir_path = 'datasets/sst'
    pheme_dir_path = 'datasets/PHEME'
    mnli_dir_path = 'datasets/NLI/MNLI'
    sick_dir_path = 'datasets/NLI/SICK'
    snli_dir_path = 'datasets/NLI/SNLI'

    sa_domains = ['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst']
    rumour_domains = ['ch', 'f', 'gw', 'os', 's']
    nli_domains = ['fiction', 'government', 'slate', 'telephone', 'travel', 'sick', 'snli']

    output_size = 1
    if target_domain in sa_domains:
        if target_domain == 'sst' or target_domain == 'imdb':
            if target_domain == 'sst':
                max_length = 128
            else:
                max_length = 196
        else:
            max_length = 128
    elif target_domain in rumour_domains:
        max_length = 64
    elif target_domain in nli_domains:
        single_sentence_max_length = 48
        max_length = single_sentence_max_length * 2
        output_size = 3

    save_dir = 'parameters/irm_{}.bin'.format(target_domain)

    weight_decay = 0.01
    lr = 1e-5  # 0.000015
    unsup_epochs = 1
    sup_epochs = 30

    if target_domain in sa_domains:
        with open('pretrained_parameters/word2id.json', 'r') as f:
            word2id = json.load(f)

        w2v = np.load('pretrained_parameters/w2v.npy')
    elif target_domain in rumour_domains:
        with open('pretrained_parameters/rumour_word2id.json', 'r') as f:
            word2id = json.load(f)

        w2v = np.load('pretrained_parameters/rumour_w2v.npy')
    elif target_domain in nli_domains:
        with open('pretrained_parameters/nli_word2id.json', 'r') as f:
            word2id = json.load(f)

        w2v = np.load('pretrained_parameters/nli_w2v.npy')

    w2v = torch.from_numpy(w2v)
    envs = []

    if target_domain in sa_domains:
        if target_domain == 'sst' or target_domain == 'imdb':
            envs = load_amazon_dataset(amazon_dir_path, target_domain, word2id, max_length, envs, device)

            if target_domain == 'imdb':
                envs = load_imdb_dataset(imdb_dir_path, envs, word2id, max_length, device)
            else:
                envs = load_sst_dataset(sst_dir_path, envs, word2id, max_length, device)

        else:
            envs = load_amazon_dataset(amazon_dir_path, target_domain, word2id, max_length, envs, device)
    elif target_domain in rumour_domains:
        envs = load_rumour_dataset(pheme_dir_path, target_domain, word2id, max_length, envs, device)
    elif target_domain in nli_domains:
        if target_domain == 'sick' or target_domain == 'snli':
            envs = load_mnli_dataset(mnli_dir_path, target_domain, word2id, single_sentence_max_length, envs, device)

            if target_domain == 'sick':
                envs = load_sick_and_nli_dataset(os.path.join(sick_dir_path, 'test.csv'), word2id,
                                                 single_sentence_max_length, envs, device)
            else:
                envs = load_sick_and_nli_dataset(os.path.join(snli_dir_path, 'test.csv'), word2id,
                                                 single_sentence_max_length, envs, device)
        else:
            envs = load_mnli_dataset(mnli_dir_path, target_domain, word2id, single_sentence_max_length, envs, device)

    model = MLP(w2v, max_length, 300, 256, output_size).to(device)

    criteria = nn.CrossEntropyLoss()

    param_list = [
        {'params': model.parameters()},
    ]

    optimizer = AdamW(
        param_list,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1.0e-8
    )

    best_macro_F1 = 0
    best_std = 0
    best_val_loss = float('inf')

    final_train_f1s = []
    final_test_f1s = []

    for epoch in range(sup_epochs):
        for env in envs:
            logits = model(env['texts'])
            if target_domain in nli_domains:
                labels = env['labels'].long()
                env['nll'] = nli_mean_nll(logits, labels.view(-1))
                env['f1'] = nli_macro_f1(logits, labels)
                env['penalty'] = nli_penalty(logits, labels, output_size)
            else:
                env['nll'] = mean_nll(logits, env['labels'])
                env['f1'] = macro_f1(logits, env['labels'])
                env['penalty'] = penalty(logits, env['labels'])

        train_nll = torch.stack([env['nll'] for env in envs[:-1]]).mean()
        train_f1 = torch.stack([env['f1'] for env in envs[:-1]]).mean()
        train_penalty = torch.stack([env['penalty'] for env in envs[:-1]]).mean()

        weight_norm = torch.tensor(0.).to(device)
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += args.l2_regularizer_weight * weight_norm
        penalty_weight = (args.penalty_weight
                          if epoch >= args.penalty_anneal_iters else 1.0)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_f1 = envs[-1]['f1']
        if epoch % 100 == 0:
            pretty_print(
                np.int32(epoch),
                train_nll.detach().cpu().numpy(),
                train_f1.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                test_f1.detach().cpu().numpy()
            )

        final_train_f1s.append(train_f1.detach().cpu().numpy())
        final_test_f1s.append(test_f1.detach().cpu().numpy())
        print('Final train acc (mean/std across restarts so far):')
        print(np.mean(final_train_f1s), np.std(final_train_f1s))
        print('Final test f1 (mean/std across restarts so far):')
        print(np.mean(final_test_f1s), np.std(final_test_f1s))

        if np.mean(final_test_f1s) > best_macro_F1:
            best_macro_F1 = np.mean(final_test_f1s)
            best_std = np.std(final_test_f1s)

    print('Best F1: %.4f, std: %.4f' % (best_macro_F1, best_std))
