'''
@Time : 2024/4/28 9:14
@Auth : Qizhi Li
'''
'''
@Time : 2024/3/20 15:12
@Auth : Qizhi Li
'''
import os
import sys
import tqdm
import time
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


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
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

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return input_ids, attention_mask, target


def read_pheme_csv(dir_path, template=None):
    data = pd.read_csv(dir_path, header=0)
    original_texts = data['texts'].tolist()
    labels = data['labels'].tolist()

    if template is not None:
        texts = []
        for text in original_texts:
            texts.append(template + text)
    else:
        texts = original_texts

    return texts, labels


def load_pheme_dataset(dir_path, target_domain, tokenizer, max_length, template):
    train_texts, train_labels, train_datasets = [], [], []

    domains = os.listdir(dir_path)
    domain_mapping = {
        'ch': 'charliehebdo',
        'f': 'ferguson',
        'gw': 'germanwings',
        'os': 'ottawashooting',
        's': 'sydneysiege',
    }

    for domain in domains:
        if domain != '{}.csv'.format(domain_mapping[target_domain]):
            texts, labels = read_pheme_csv(os.path.join(dir_path, domain), template)
            train_dataset = CustomDataset(texts, labels, tokenizer, max_length)
            train_datasets.append(train_dataset)

    texts, labels = read_pheme_csv(os.path.join(dir_path, '{}.csv'.format(domain_mapping[target_domain])), template)
    test_dataset = CustomDataset(texts, labels, tokenizer, max_length)

    return train_datasets, test_dataset


def read_imdb_csv(dir_path, template=None):
    data = pd.read_csv(dir_path, header=0)
    original_texts = data['sentence'].tolist()
    labels = data['labels'].tolist()

    if template is not None:
        texts = []
        for text in original_texts:
            texts.append(template + text)
    else:
        texts = original_texts

    return texts, labels


def load_imdb_dataset(dir_path, tokenizer, max_length, template=None):
    val_texts, val_labels = read_imdb_csv(os.path.join(dir_path, 'dev.csv'), template)
    test_texts, test_labels = read_imdb_csv(os.path.join(dir_path, 'test.csv'), template)

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


def read_sst_tsv(dir_path, template=None):
    data = pd.read_csv(dir_path, sep='\t', header=0)
    original_texts = data['sentence'].tolist()
    labels = data['label'].tolist()

    if template is not None:
        texts = []
        for text in original_texts:
            texts.append(template + text)
    else:
        texts = original_texts

    return texts, labels


def load_sst_dataset(dir_path, tokenizer, max_length, template=None):
    val_texts, val_labels = read_sst_tsv(os.path.join(dir_path, 'dev.tsv'), template)
    test_texts, test_labels = read_sst_tsv(os.path.join(dir_path, 'test.tsv'), template)

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


def load_amazon_dataset(dir_path, target_domain, tokenizer, max_length, template=None):
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
                    if template is not None:
                        train_texts.append(template + line[0])
                    else:
                        train_texts.append(line[0])

                    train_labels.append(line[1])

    for num_samples in train_domains:
        train_dataset = CustomDataset(train_texts[:num_samples], train_labels[:num_samples],
                                      tokenizer, max_length)
        del train_texts[:num_samples]
        del train_labels[:num_samples]
        train_datasets.append(train_dataset)

    if target_domain in domains:
        with open(os.path.join(dir_path, target_domain, 'test.txt'), 'r') as f:
            lines = f.readlines()

        for text in lines:
            line = text.strip().split(' ||| ')
            if len(line) == 2:
                if template is not None:
                    test_texts.append(template + line[0])
                else:
                    test_texts.append(line[0])
                test_labels.append(line[1])

        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

        return train_datasets, test_dataset
    else:
        return train_datasets


def read_nli_csv(dir_path, template, tokenizer, single_sentence_max_length):
    df = pd.read_csv(dir_path, header=0)
    sentence1 = df['sentence1'].tolist()
    sentence2 = df['sentence2'].tolist()
    labels = df['label'].tolist()

    texts = []
    for i in range(len(sentence1)):
        # If the text is too long, truncate the text while removing the [CLS] [SEP]
        sentence1_truncation = tokenizer.decode(tokenizer.encode(sentence1[i],
                                                                 truncation=True,
                                                                 max_length=single_sentence_max_length)[1: -1])
        sentence2_truncation = tokenizer.decode(tokenizer.encode(sentence2[i],
                                                                 truncation=True,
                                                                 max_length=single_sentence_max_length)[1: -1])

        text = template.replace("[sentence1]", '\'{}\''.format(sentence1_truncation)).replace("[sentence2]",
                                                                                              '\'{}\''.format(
                                                                                                  sentence2_truncation))
        texts.append(text)

    return texts, labels


def load_mnli_datasets(dir_path, target_domain, tokenizer, single_sentence_max_length, max_length, template):
    train_datasets = []
    domains = os.listdir(dir_path)

    for domain in domains:
        if domain != target_domain:
            texts, labels = read_nli_csv(os.path.join(dir_path, domain, 'train.csv'),
                                         template, tokenizer, single_sentence_max_length)
            train_dataset = CustomDataset(texts, labels, tokenizer, max_length)
            train_datasets.append(train_dataset)

    if target_domain in domains:
        test_texts, test_labels = read_nli_csv(os.path.join(dir_path, target_domain, 'test.csv'),
                                               template, tokenizer, single_sentence_max_length)

        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

        return train_datasets, test_dataset
    else:
        return train_datasets


def load_snli_sick_datasets(dir_path, tokenizer, single_sentence_max_length, max_length, template):
    val_texts, val_labels = read_nli_csv(os.path.join(dir_path, 'dev.csv'),
                                         template, tokenizer, single_sentence_max_length)
    test_texts, test_labels = read_nli_csv(os.path.join(dir_path, 'test.csv'),
                                           template, tokenizer, single_sentence_max_length)

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


def count_parameters(model):
    num_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad and 'roberta' not in name and 'lm_head' not in name:
            num_parameters += param.numel()
    return num_parameters / 1e6  # Convert the parameter quantity to millions (M)


def obtain_verbalizer_ids(verbalizer, tokenizer):
    """
    Convert the words in the verbalizer into the ids of the Embedding layer
    :param verbalizer: dict
    :param tokenizer: Object
    :return verbalizer_ids: list
            The ids of all words in the verbalizer in the Embedding layer
    :return index2ids: dict
            The mapping between index and token id in verbalizer_ids
    """
    verbalizer_ids = tokenizer.convert_tokens_to_ids(list(verbalizer.keys()))
    index2ids = {i: verbalizer_ids[i] for i in range(len(verbalizer_ids))}
    return verbalizer_ids, index2ids


def evaluate(model, data_loader, tgt_domain):
    preds = []
    labels = []
    loss_total = 0.0
    model.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc='test on {}'.format(tgt_domain)):
            input_ids = batch[0].to(device)
            att_masks = batch[1].to(device)
            tgt = batch[2].to(device)
            labels.extend(tgt.cpu().tolist())

            logits = model(input_ids, att_masks)

            pred = torch.argmax(logits, dim=-1)
            loss = F.cross_entropy(logits, tgt)
            loss_total += loss
            preds.extend(pred.cpu().detach().tolist())

    model.train()
    macro_F1 = f1_score(labels, preds, average='macro')

    return loss_total / len(data_loader), macro_F1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_domain',
                        choices=['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst',
                                 'ch', 'f', 'gw', 'os', 's', 'fiction', 'government',
                                 'slate', 'telephone', 'travel', 'sick', 'snli'],
                        default='book')
    parser.add_argument('--method', choices=['mlm', 'cls'], default='mlm')
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--cuda', default=0)
    args = parser.parse_args()

    sa_domains = ['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst']
    rumour_domains = ['ch', 'f', 'gw', 'os', 's']
    nli_domains = ['fiction', 'government', 'slate', 'telephone', 'travel', 'sick', 'snli']
    print(args)

    seed = args.seed
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    model_path = '/sdc1/liqizhi/huggingface/roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    amazon_dir_path = 'datasets/amazon'
    imdb_dir_path = 'datasets/imdb'
    sst_dir_path = 'datasets/sst'
    pheme_dir_path = 'datasets/PHEME'
    mnli_dir_path = 'datasets/NLI/MNLI'
    sick_dir_path = 'datasets/NLI/SICK'
    snli_dir_path = 'datasets/NLI/SNLI'

    target_domain = args.target_domain

    if args.method == 'mlm':
        if target_domain in nli_domains:
            template = '\'[sentence1]\'? <mask>, \'[sentence2]\''
        else:
            template = 'It was <mask>. '
    else:
        template = None

    if target_domain in sa_domains:
        verbalizer = {
            'good': 1,
            'bad': 0
        }
    elif target_domain in rumour_domains:
        verbalizer = {
            'rumour': 1,
            'truth': 0
        }
    elif target_domain in nli_domains:
        verbalizer = {
            'uncertain': 0,
            'no': 1,
            'yes': 2,
        }

    if target_domain == 'imdb':
        max_length = 196
    elif target_domain in sa_domains:
        max_length = 128
    elif target_domain in rumour_domains:
        max_length = 64
    elif target_domain in nli_domains:
        single_sentence_max_length = 48
        max_length = 128

    batch_size = 16
    weight_decay = 1e-2
    lr = 1e-5  # 0.000015
    num_epochs = 30

    set_seed(seed)

    if target_domain in sa_domains:
        if target_domain != 'sst' and target_domain != 'imdb':
            train_datasets, test_dataset = load_amazon_dataset(amazon_dir_path, target_domain, tokenizer,
                                                               max_length, template)
        else:
            train_datasets = load_amazon_dataset(amazon_dir_path, target_domain, tokenizer, max_length, template)

            if args.target_domain == 'imdb':
                _, test_dataset = load_imdb_dataset(imdb_dir_path, tokenizer, max_length, template)
            else:
                _, test_dataset = load_sst_dataset(sst_dir_path, tokenizer, max_length, template)

    elif target_domain in rumour_domains:
        train_datasets, test_dataset = load_pheme_dataset(pheme_dir_path, target_domain, tokenizer, max_length,
                                                          template)
    elif target_domain in nli_domains:
        if target_domain == 'sick':
            train_datasets = load_mnli_datasets(mnli_dir_path, target_domain, tokenizer,
                                                single_sentence_max_length, max_length, template)
            _, test_dataset = load_snli_sick_datasets(sick_dir_path, tokenizer,
                                                                single_sentence_max_length,
                                                                max_length, template)
        elif target_domain == 'snli':
            train_datasets = load_mnli_datasets(mnli_dir_path, target_domain, tokenizer,
                                                single_sentence_max_length, max_length, template)
            _, test_dataset = load_snli_sick_datasets(snli_dir_path, tokenizer,
                                                                single_sentence_max_length,
                                                                max_length, template)
        else:
            train_datasets, test_dataset = load_mnli_datasets(mnli_dir_path, target_domain, tokenizer,
                                                              single_sentence_max_length, max_length, template)

    train_dataloader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
                        for train_dataset in train_datasets]
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    verbalizer_ids, index2ids = obtain_verbalizer_ids(verbalizer, tokenizer)

    if args.method == 'mlm':
        model = GenPromptCL.BaselineMlmModel(model_path, tokenizer, verbalizer_ids).to(device)
    else:
        model = GenPromptCL.BaselineClsModel(model_path).to(device)

    criteria = nn.CrossEntropyLoss()

    model_parameters = count_parameters(model)
    print('Number of parameters: {:.4f}M'.format(model_parameters))

    mean_sort = 'full'
    mean_decay_param = 1.0
    cov_weighting_loss = GenPromptCL.CoVWeightingLoss(mean_sort, mean_decay_param, device)

    param_list = [
        {'params': model.parameters()},
    ]

    optimizer = optim.AdamW(
        param_list,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1.0e-8
    )

    num_batches = 0
    for loader in train_dataloader:
        num_batches += len(loader)

    best_F1 = 0.0
    best_val_loss = float('inf')
    index = 0
    last_improve = 0
    require_improvement = 5
    flag = False
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_train_steps = 0

        pbar = tqdm.tqdm(zip(*train_dataloader), desc='Seed {}, Training epoch {}'.format(seed, epoch))
        for step, batches in enumerate(pbar):
            model.train()

            logits_list, labels = [], []

            for idx, batch in enumerate(batches):
                input_ids = batch[0].to(device)
                att_masks = batch[1].to(device)
                tgt = batch[2].to(device)
                logits = model(input_ids, att_masks)
                logits_list.append(logits)
                labels.append(tgt)

            logits = torch.cat(logits_list, dim=0)
            labels = torch.cat(labels, dim=0).to(device)
            loss = criteria(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            num_train_steps += 1

        _, f1 = evaluate(model, test_dataloader, target_domain)

        if f1 > best_F1:
            best_F1 = f1
            last_improve = index
        print("Epoch {}, F1: {:.4f}, ".format(epoch, f1))

    if not os.path.exists('results/scores/baseline'):
        os.makedirs('results/scores/baseline')

    with open(f'results/scores/baseline/{target_domain}.txt', 'a') as f:
        f.write(f'seed: {seed} | target: {target_domain} | f1: {best_F1}\n')
