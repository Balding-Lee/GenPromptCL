'''
@Time : 2024/3/19 16:14
@Auth : Qizhi Li
'''
import os
import re
import math
import nltk
import tqdm
import json
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn import svm
from torch.optim import SGD
import torch.nn.functional as F
from typing import Optional, List
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


class AverageMeter(object):
    r"""Computes and stores the average and current value.

    Examples::

        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeterDict(object):
    def __init__(self, names: List, fmt: Optional[str] = ':f'):
        self.dict = {
            name: AverageMeter(name, fmt) for name in names
        }

    def reset(self):
        for meter in self.dict.values():
            meter.reset()

    def update(self, accuracies, n=1):
        for name, acc in accuracies.items():
            self.dict[name].update(acc, n)

    def average(self):
        return {
            name: meter.avg for name, meter in self.dict.items()
        }

    def __getitem__(self, item):
        return self.dict[item]


class Meter(object):
    """Computes and stores the current value."""

    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def load_amazon_data(domain):
    sentences = []
    with open(os.path.join('datasets/amazon', domain, 'all_data.txt'), 'r') as f:
        texts = f.readlines()

    for i in tqdm.trange(len(texts), desc='loading {} data'.format(domain)):
        line = texts[i].strip().split(' ||| ')
        if len(line) == 2:
            sentences.append(line[0])

    return sentences


def load_imdb_data(file_name):
    dir_path = 'datasets/imdb/{}.csv'.format(file_name)
    data = pd.read_csv(dir_path, header=0)
    texts = data['sentence'].tolist()

    return texts


def load_sst_data(file_name):
    dir_path = 'datasets/sst/{}.tsv'.format(file_name)
    data = pd.read_csv(dir_path, sep='\t', header=0)
    texts = data['sentence'].tolist()

    return texts


def load_rumour_data(domain):
    dir_path = 'datasets/PHEME/{}.csv'.format(domain)
    df = pd.read_csv(dir_path, header=0)
    texts = df['texts'].tolist()
    return texts


def load_nli_data(file_path):
    df = pd.read_csv(file_path, header=0)
    sentence1 = df['sentence1'].tolist()
    sentence2 = df['sentence2'].tolist()

    texts = []
    # for s1, s2 in zip(sentence1, sentence2):
    #     texts.append(s1 + ' ' + s2)
    texts.extend(sentence1)
    texts.extend(sentence2)

    return texts


def prepare_word2vec(path):
    w2v = KeyedVectors.load_word2vec_format(path, binary=True)

    nltk.download('punkt')

    sentences = []
    amazon_domains = ['book', 'dvd', 'electronics', 'kitchen']
    movie_file_names = ['dev', 'test']
    rumour_domains = ['charliehebdo', 'ferguson', 'germanwings', 'ottawashooting', 'sydneysiege']
    nli_domains = ['MNLI/fiction', 'MNLI/government', 'MNLI/slate', 'MNLI/telephone', 'MNLI/travel', 'SICK', 'SNLI']

    # for domain in amazon_domains:
    #     sentences.extend(load_amazon_data(domain))
    #
    # for file_name in movie_file_names:
    #     sentences.extend(load_imdb_data(file_name))
    #     sentences.extend(load_sst_data(file_name))

    # for domain in rumour_domains:
    #     sentences.extend(load_rumour_data(domain))

    for nli_domain in nli_domains:
        if '/' in nli_domain:
            sentences.extend(load_nli_data(os.path.join('datasets/NLI', nli_domain, 'train.csv')))
            sentences.extend(load_nli_data(os.path.join('datasets/NLI', nli_domain, 'test.csv')))
        else:
            sentences.extend(load_nli_data(os.path.join('datasets/NLI', nli_domain, 'test.csv')))

    words = set()

    for i in tqdm.trange(len(sentences), desc='tokenize sentences'):
        try:
            s_words = word_tokenize(sentences[i])
            words.update(s_words)
        except:
            print(sentences[i])
            break

    i, word2id, id2word, embeddings = 0, {}, {}, []
    for word in tqdm.tqdm(words, desc='finding word embeddings'):
        word2id[word] = i
        id2word[i] = word
        if w2v.__contains__(word):
            embeddings.append(w2v[word])
        else:
            embeddings.append(w2v['UNK'])
        i += 1

    word2id['<PAD>'] = i
    id2word[i] = '<PAD>'
    embeddings.append(w2v['PAD'])

    with open('pretrained_parameters/nli_word2id.json', 'w') as f:
        json.dump(word2id, f)

    with open('pretrained_parameters/nli_id2word.json', 'w') as f:
        json.dump(id2word, f)

    embeddings = np.array(embeddings)
    np.save('pretrained_parameters/nli_w2v.npy', embeddings)


def prepare_NLI_dataset():
    # MNLI
    MNLI_file_path = 'datasets/NLI/MNLI'
    MNLI_files = os.listdir(MNLI_file_path)
    for MNLI_file in tqdm.tqdm(MNLI_files, desc='preparing MNLI'):
        file_names = ['train', 'dev', 'test']
        for file_name in file_names:
            sentence1, sentence2 = [], []
            with open(os.path.join(MNLI_file_path, MNLI_file, file_name), 'rb') as f:
                data = pickle.load(f)

            assert len(data[0]) == len(data[1])
            for i in range(len(data[0])):
                sentence1.append(data[0][i][0])
                sentence2.append(data[0][i][1])

            label = data[1]

            df = pd.DataFrame({'sentence1': sentence1, 'sentence2': sentence2, 'label': label})
            df.to_csv(os.path.join(MNLI_file_path, MNLI_file, '{}.csv'.format(file_name)), index=False)

    # SICK
    with open(r'D:\数据集\SICK\SICK_annotated.txt', 'r') as f:
        data = f.readlines()

    train_sentence1, train_sentence2, train_label = [], [], []
    dev_sentence1, dev_sentence2, dev_label = [], [], []
    test_sentence1, test_sentence2, test_label = [], [], []
    label_mapping = {
        'NEUTRAL': 0,
        'CONTRADICTION': 1,
        'ENTAILMENT': 2,
    }
    for i in tqdm.trange(len(data), desc='preparing SICK'):
        if i == 0:
            continue

        split_data = data[i].rstrip('\n').split('\t')
        if split_data[-1] == 'TRAIN':
            train_sentence1.append(split_data[2])
            train_sentence2.append(split_data[4])
            train_label.append(label_mapping[split_data[7]])
        elif split_data[-1] == 'TRIAL':
            dev_sentence1.append(split_data[2])
            dev_sentence2.append(split_data[4])
            dev_label.append(label_mapping[split_data[7]])
        elif split_data[-1] == 'TEST':
            test_sentence1.append(split_data[2])
            test_sentence2.append(split_data[4])
            test_label.append(label_mapping[split_data[7]])

    train_df = pd.DataFrame({'sentence1': train_sentence1, 'sentence2': train_sentence2, 'label': train_label})
    dev_df = pd.DataFrame({'sentence1': dev_sentence1, 'sentence2': dev_sentence2, 'label': dev_label})
    test_df = pd.DataFrame({'sentence1': test_sentence1, 'sentence2': test_sentence2, 'label': test_label})

    train_df.to_csv('datasets/NLI/SICK/train.csv', index=False)
    dev_df.to_csv('datasets/NLI/SICK/dev.csv', index=False)
    test_df.to_csv('datasets/NLI/SICK/test.csv', index=False)

    # SNLI
    with open(r'D:\数据集\SNLI\dataset.jsonl', 'r') as f:
        data = f.readlines()

    sentence1, sentence2, labels = [], [], []
    label_mapping = {
        'neutral': 0,
        'contradiction': 1,
        'entailment': 2,
    }
    for i in tqdm.trange(len(data), desc='preparing SNLI'):
        data_dict = json.loads(data[i].rstrip('\n'))
        sentence1.append(data_dict['sentence1'])
        sentence2.append(data_dict['sentence2'])
        labels.append(label_mapping[data_dict['gold_label']])

    dev_sentence1, test_sentence1, dev_sentence2, test_sentence2, dev_label, test_label = train_test_split(
        sentence1, sentence2, labels, test_size=0.5, random_state=42
    )

    dev_df = pd.DataFrame({'sentence1': dev_sentence1, 'sentence2': dev_sentence2, 'label': dev_label})
    test_df = pd.DataFrame({'sentence1': test_sentence1, 'sentence2': test_sentence2, 'label': test_label})

    dev_df.to_csv('datasets/NLI/SNLI/dev.csv', index=False)
    test_df.to_csv('datasets/NLI/SNLI/test.csv', index=False)


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x


def calculate(source_feature: torch.Tensor, target_feature: torch.Tensor,
              device, progress=True, training_epochs=10):
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.

    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier

    Returns:
        :math:`\mathcal{A}`-distance
    """
    source_label = torch.ones((source_feature.shape[0], 1))
    target_label = torch.zeros((target_feature.shape[0], 1))
    feature = torch.cat([source_feature, target_feature], dim=0)
    label = torch.cat([source_label, target_label], dim=0)

    dataset = TensorDataset(feature, label)
    length = len(dataset)
    train_size = int(0.8 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    anet = ANet(feature.shape[1]).to(device)
    optimizer = SGD(anet.parameters(), lr=0.01)
    # a_distance = 2.0
    best_acc = float('inf')
    out_acc = 0.0
    for epoch in range(training_epochs):
        anet.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)
            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()

        anet.eval()
        meter = AverageMeter("accuracy", ":4.2f")
        accs = []
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y = anet(x)
                acc = binary_accuracy(y, label)
                # meter.update(acc, x.shape[0])
                accs.append(acc.detach().cpu().numpy())

        mean_acc = np.mean(np.array(accs))
        # error = 1 - meter.avg / 100
        # a_distance = 2 * (1 - 2 * error)
        if abs(mean_acc - 50) < best_acc:
            best_acc = abs(mean_acc - 50)
            if mean_acc > 50:
                out_acc = mean_acc
            else:
                out_acc = 50 + abs(mean_acc - 50)
        if progress:
            # print("epoch {} accuracy: {}".format(epoch, meter.avg))
            print("epoch {} accuracy: {} out acc: {}".format(epoch, mean_acc, out_acc))

    error = 1 - out_acc / 100
    a_distance = 2 * (1 - 2 * error)

    return a_distance


def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source / 2), int(nb_target / 2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)


def gaussian_kernel(X, Y, sigma=1.0):
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)

    X_sqnorms = np.diag(XX)
    Y_sqnorms = np.diag(YY)

    K = np.exp(-0.5 * (X_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * XY) / sigma ** 2)
    return K


def mmd_loss(X, Y, sigma=15):
    # 15
    K_XX = gaussian_kernel(X, X, sigma=sigma)
    K_YY = gaussian_kernel(Y, Y, sigma=sigma)
    K_XY = gaussian_kernel(X, Y, sigma=sigma)

    m = X.shape[0]
    n = Y.shape[0]

    mmd = (np.sum(K_XX) / (m * (m - 1)) +
           np.sum(K_YY) / (n * (n - 1)) -
           2 * np.sum(K_XY) / (m * n))

    return mmd


def prepare_PHEME():
    folders = ['charliehebdo-all-rnr-threads',
               'ferguson-all-rnr-threads',
               'germanwings-crash-all-rnr-threads',
               'ottawashooting-all-rnr-threads',
               'sydneysiege-all-rnr-threads']

    dir_path = r'D:\数据集\PHEME\all-rnr-annotated-threads'

    for folder in tqdm.tqdm(folders):
        texts = []
        labels = []

        rumour_path = os.path.join(dir_path, folder, 'rumours')
        non_rumour_path = os.path.join(dir_path, folder, 'non-rumours')

        files = os.listdir(rumour_path)

        for file in files:
            if not file.startswith('.'):
                json_file_path = os.path.join(rumour_path, file, 'source-tweets', '{}.json'.format(file))
                with open(json_file_path, 'r') as f:
                    text = json.load(f)['text']

                text = re.sub(r'[\t\n]', '', text)
                texts.append(text)
                labels.append(1)

        files = os.listdir(non_rumour_path)

        for file in files:
            if not file.startswith('.'):
                json_file_path = os.path.join(non_rumour_path, file, 'source-tweets', '{}.json'.format(file))
                with open(json_file_path, 'r') as f:
                    text = json.load(f)['text'].replace('\t', '').replace('\n', '')

                text = re.sub(r'[\t\n]', '', text)
                texts.append(text)
                labels.append(0)

        df = pd.DataFrame({'texts': texts, 'labels': labels})
        df.to_csv('./datasets/PHEME/{}.csv'.format(folder.split('-')[0]), index=False)


def compute_mmd(x, y, sigma=1.0):
    """
    计算MMD (Maximum Mean Discrepancy)
    :param x: tensor, shape (batch_size, feature_dim),
    :param y: tensor, shape (batch_size, feature_dim),
    :param sigma: float,
    :return:
    """

    # 计算高斯核
    def rbf_kernel(x, y, sigma):
        x_norm = torch.sum(x ** 2, dim=1).view(-1, 1)  # shape: (batch_size, 1)
        y_norm = torch.sum(y ** 2, dim=1).view(1, -1)  # shape: (1, batch_size)
        dist = x_norm + y_norm - 2 * torch.mm(x, y.t())  # shape: (batch_size, batch_size)
        return torch.exp(-dist / (2 * sigma ** 2))

    # 计算MMD
    xx_kernel = rbf_kernel(x, x, sigma)
    yy_kernel = rbf_kernel(y, y, sigma)
    xy_kernel = rbf_kernel(x, y, sigma)

    mmd = torch.mean(xx_kernel) + torch.mean(yy_kernel) - 2 * torch.mean(xy_kernel)

    return mmd

