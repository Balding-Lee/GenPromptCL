'''
@Time : 2024/5/7 16:02
@Auth : Qizhi Li
'''
import os
import sys
import tqdm
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader

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


class CustomDataset(Dataset):
    def __init__(self, data, targets, tokenizer, max_length):
        self.data = data
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        target = int(self.targets[index])

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 转换为一维向量
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return input_ids, attention_mask, target


def read_imdb_csv(dir_path):
    data = pd.read_csv(dir_path, header=0)
    texts = data['sentence'].tolist()
    labels = data['labels'].tolist()

    return texts, labels


def load_imdb_dataset(dir_path, tokenizer, max_length):
    test_texts, test_labels = read_imdb_csv(os.path.join(dir_path, 'test.csv'))

    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return test_dataset


def read_sst_tsv(dir_path):
    data = pd.read_csv(dir_path, sep='\t', header=0)
    texts = data['sentence'].tolist()
    labels = data['label'].tolist()

    return texts, labels


def load_sst_dataset(dir_path, tokenizer, max_length):
    test_texts, test_labels = read_sst_tsv(os.path.join(dir_path, 'test.tsv'))

    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)
    return test_dataset


def load_amazon_dataset(dir_path, target_domain, tokenizer, max_length):
    train_texts, train_labels, train_domains, train_datasets = [], [], [], []
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
                    train_labels.append(line[1])

    for num_samples in train_domains:
        texts = train_texts[:num_samples]
        labels = train_labels[:num_samples]
        train_dataset = CustomDataset(texts, labels, tokenizer, max_length)
        del train_texts[:num_samples]
        del train_labels[:num_samples]
        train_datasets.append(train_dataset)

    if target_domain in domains:
        with open(os.path.join(dir_path, target_domain, 'test.txt'), 'r') as f:
            lines = f.readlines()

        for text in lines:
            line = text.strip().split(' ||| ')
            if len(line) == 2:
                test_texts.append(line[0])
                test_labels.append(line[1])

        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

        return train_datasets, test_dataset
    else:
        return train_datasets


def read_pheme_csv(dir_path):
    data = pd.read_csv(dir_path, header=0)
    texts = data['texts'].tolist()
    labels = data['labels'].tolist()
    return texts, labels


def load_rumour_dataset(dir_path, target_domain, tokenizer, max_length):
    train_texts, train_labels, train_domains, train_datasets = [], [], [], []

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
            train_dataset = CustomDataset(texts, labels, tokenizer, max_length)
            train_datasets.append(train_dataset)

    test_texts, test_labels = read_pheme_csv(os.path.join(dir_path, '{}.csv'.format(domain_mapping[target_domain])))
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return train_datasets, test_dataset


def read_nli_csv(dir_path, tokenizer, single_sentence_max_length):
    df = pd.read_csv(dir_path, header=0)
    sentence1 = df['sentence1'].tolist()
    sentence2 = df['sentence2'].tolist()
    labels = df['label'].tolist()

    texts = []
    for i in range(len(sentence1)):
        sentence1_truncation = tokenizer.decode(tokenizer.encode(sentence1[i],
                                                                 truncation=True,
                                                                 max_length=single_sentence_max_length)[1: -1])
        sentence2_truncation = tokenizer.decode(tokenizer.encode(sentence2[i],
                                                                 truncation=True,
                                                                 max_length=single_sentence_max_length)[1: -1])

        text = sentence1_truncation + tokenizer.sep_token + sentence2_truncation
        texts.append(text)

    return texts, labels


def load_mnli_datasets(dir_path, target_domain, tokenizer, single_sentence_max_length, max_length):
    train_datasets, unsup_train_texts = [], []
    domains = os.listdir(dir_path)

    for domain in domains:
        if domain != target_domain:
            texts, labels = read_nli_csv(os.path.join(dir_path, domain, 'train.csv'),
                                         tokenizer, single_sentence_max_length)
            train_dataset = CustomDataset(texts, labels, tokenizer, max_length)
            train_datasets.append(train_dataset)

    if target_domain in domains:
        test_texts, test_labels = read_nli_csv(os.path.join(dir_path, target_domain, 'test.csv'),
                                               tokenizer, single_sentence_max_length)

        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

        return train_datasets, test_dataset
    else:
        return train_datasets


def load_snli_sick_datasets(dir_path, tokenizer, single_sentence_max_length, max_length):
    test_texts, test_labels = read_nli_csv(os.path.join(dir_path, 'test.csv'),
                                           tokenizer, single_sentence_max_length)

    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return test_dataset


def count_parameters(model):
    num_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad and 'roberta' not in name and 'lm_head' not in name:
            num_parameters += param.numel()
    return num_parameters / 1e6


def test(model, dataset_loader, e, device):
    model.eval()
    test_loss = 0
    y_true = []
    y_pred = []
    for input_ids, att_masks, target in dataset_loader:
        input_ids, att_masks, target = input_ids.to(device), att_masks.to(device), target.to(device)

        out, _ = model(input_ids, att_masks, input_ids, att_masks)

        # sum up batch loss
        test_loss += torch.nn.functional.cross_entropy(out, target, size_average=False).item()

        # get the index of the max log-probability
        pred = out.data.max(1, keepdim=True)[1]
        y_true.extend(target.cpu().numpy())
        y_pred.extend(pred.cpu().numpy().flatten())

    test_loss /= len(dataset_loader.dataset)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    return {
        'epoch': e,
        'average_loss': test_loss,
        'total': len(dataset_loader.dataset),
        'macro_f1': macro_f1
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_domain',
                        choices=['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst',
                                 'ch', 'f', 'gw', 'os', 's', 'fiction', 'government',
                                 'slate', 'telephone', 'travel', 'sick', 'snli'],
                        default='book')
    parser.add_argument('--seed', default=9)
    parser.add_argument('--cuda', default=0)
    args = parser.parse_args()

    seed = args.seed
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    target_domain = args.target_domain

    set_seed(seed)

    model_path = '/home/liqizhi/huggingface/roberta-base'
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

    save_dir = 'parameters/deepcoral_{}.bin'.format(target_domain)

    num_outputs = 2
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
        max_length = 128
        single_sequence_max_length = 48
        num_outputs = 3

    backbone_save_dir = 'parameters/simcse_{}_{}.bin'.format(target_domain, seed)
    classifier_save_dir = 'parameters/simcse_cls_{}_{}.bin'.format(target_domain, seed)

    batch_size = 16
    weight_decay = 0.01
    lr = 1e-3  # 0.000015
    epochs = 30

    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    if target_domain in sa_domains:
        if target_domain == 'sst' or target_domain == 'imdb':
            train_datasets = load_amazon_dataset(amazon_dir_path, target_domain, tokenizer, max_length)

            if target_domain == 'imdb':
                test_dataset = load_imdb_dataset(imdb_dir_path, tokenizer, max_length)
            else:
                test_dataset = load_sst_dataset(sst_dir_path, tokenizer, max_length)

        else:
            train_datasets, test_dataset = load_amazon_dataset(amazon_dir_path, target_domain,
                                                               tokenizer, max_length)
    elif target_domain in rumour_domains:
        train_datasets, test_dataset = load_rumour_dataset(pheme_dir_path, target_domain, tokenizer, max_length)
    elif target_domain in nli_domains:
        if target_domain == 'sick' or target_domain == 'snli':
            train_datasets = load_mnli_datasets(mnli_dir_path, target_domain, tokenizer,
                                                single_sequence_max_length, max_length)
            if target_domain == 'sick':
                test_dataset = load_snli_sick_datasets(sick_dir_path, tokenizer, single_sequence_max_length,
                                                       max_length)
            else:
                test_dataset = load_snli_sick_datasets(snli_dir_path, tokenizer, single_sequence_max_length,
                                                       max_length)
        else:
            train_datasets, test_dataset = load_mnli_datasets(mnli_dir_path, target_domain, tokenizer,
                                                              single_sequence_max_length, max_length)

    train_dataloader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        for train_dataset in train_datasets]
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GenPromptCL.DeepCORAL(model_path, num_outputs).to(device)

    model_parameters = count_parameters(model)
    print('Number of parameters: {:.4f}M'.format(model_parameters))

    criteria = nn.CrossEntropyLoss()

    param_list = [
        {'params': model.backbone.parameters()},
        {'params': model.fc.parameters(), 'lr': 10 * lr}
    ]

    optimizer = optim.SGD(
        param_list,
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9
    )

    num_batches = 0
    for loader in train_dataloader:
        num_batches += len(loader)

    best_macro_F1 = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        result = []
        _lambda = (epoch + 1) / epochs

        sources = [list(enumerate(source_loader)) for source_loader in train_dataloader]
        target = list(enumerate(test_dataloader))

        train_steps = float('inf')
        for source in sources:
            if len(source) < train_steps:
                train_steps = len(source)

        train_steps = min(train_steps, len(target))

        for batch_idx in tqdm.trange(train_steps):

            for source in sources:
                _, (source_input_ids, source_att_masks, source_label) = source[batch_idx]
                _, (target_input_ids, target_att_masks, _) = target[batch_idx]

                source_input_ids = source_input_ids.to(device)
                source_att_masks = source_att_masks.to(device)
                source_label = source_label.to(device)
                target_input_ids = target_input_ids.to(device)
                target_att_masks = target_att_masks.to(device)

                out1, out2 = model(source_input_ids, source_att_masks,
                                   target_input_ids, target_att_masks)

                classification_loss = criteria(out1, source_label)
                coral_loss = GenPromptCL.CORAL(out1, out2)

                loss = _lambda * coral_loss + classification_loss

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

        test_target = test(model, test_dataloader, epoch, device)

        print('###Test Target {}: Epoch: {}, avg_loss: {:.4f}, macro-F1: {:.4f}'.format(
            target_domain,
            epoch + 1,
            test_target['average_loss'],
            test_target['macro_f1'],
        ))

        if test_target['macro_f1'] > best_macro_F1:
            best_macro_F1 = test_target['macro_f1']
            torch.save(model.state_dict(), save_dir)

    print('Best macro-F1: %.4f' % best_macro_F1)
