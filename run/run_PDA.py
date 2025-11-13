import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from models.PDA import MaksedLanguageModel, DomainDiscriminators, DomainKL


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


def read_pheme_csv(dir_path, template, i=None):
    data = pd.read_csv(dir_path, header=0)
    original_texts = data['texts'].tolist()
    labels = data['labels'].tolist()
    domain_labels = []

    texts = []
    for text in original_texts:
        texts.append(template + text)
        if i is not None:
            domain_labels.append(i)

    if i is not None:
        return texts, labels, domain_labels
    else:
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

    i = 0  # domain 的标签
    for domain in domains:
        if domain != '{}.csv'.format(domain_mapping[target_domain]):
            texts, labels, train_domain_labels = read_pheme_csv(os.path.join(dir_path, domain), template, i)
            train_dataset = CustomDataset(texts, labels, tokenizer, max_length)
            train_datasets.append(train_dataset)
            i += 1

    texts, labels = read_pheme_csv(os.path.join(dir_path, '{}.csv'.format(domain_mapping[target_domain])), template)
    test_dataset = CustomDataset(texts, labels, tokenizer, max_length)

    return train_datasets, test_dataset


def read_imdb_csv(dir_path, template):
    data = pd.read_csv(dir_path, header=0)
    original_texts = data['sentence'].tolist()
    labels = data['labels'].tolist()

    texts = []
    for text in original_texts:
        texts.append(template + text)

    return texts, labels


def load_imdb_dataset(dir_path, tokenizer, max_length, template):
    val_texts, val_labels = read_imdb_csv(os.path.join(dir_path, 'dev.csv'), template)
    test_texts, test_labels = read_imdb_csv(os.path.join(dir_path, 'test.csv'), template)

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


def read_sst_tsv(dir_path, template):
    data = pd.read_csv(dir_path, sep='\t', header=0)
    original_texts = data['sentence'].tolist()
    labels = data['label'].tolist()

    texts = []
    for text in original_texts:
        texts.append(template + text)

    return texts, labels


def load_sst_dataset(dir_path, tokenizer, max_length, template):
    val_texts, val_labels = read_sst_tsv(os.path.join(dir_path, 'dev.tsv'), template)
    test_texts, test_labels = read_sst_tsv(os.path.join(dir_path, 'test.tsv'), template)

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


def load_amazon_dataset(dir_path, target_domain, template, tokenizer, max_length):
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
                    train_texts.append(template + line[0])
                    train_labels.append(line[1])

    for num_samples in train_domains:
        train_dataset = CustomDataset(train_texts[:num_samples], train_labels[:num_samples], tokenizer, max_length)
        del train_texts[:num_samples]
        del train_labels[:num_samples]
        train_datasets.append(train_dataset)

    if target_domain in domains:
        with open(os.path.join(dir_path, target_domain, 'test.txt'), 'r') as f:
            lines = f.readlines()

        for text in lines:
            line = text.strip().split(' ||| ')
            if len(line) == 2:
                test_texts.append(template + line[0])
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


def obtain_verbalizer_ids(verbalizer, tokenizer):
    verbalizer_ids = tokenizer.convert_tokens_to_ids(list(verbalizer.keys()))
    index2ids = {i: verbalizer_ids[i] for i in range(len(verbalizer_ids))}
    return verbalizer_ids, index2ids


def count_parameters(model):
    num_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad and 'roberta' not in name and 'lm_head' not in name:
            num_parameters += param.numel()
    return num_parameters / 1e6  # 将参数量转换为百万（M）


def evaluate(model, data_loader, tgt_domain):
    preds = []
    labels = []
    loss_total = 0.0
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='test on {}'.format(tgt_domain)):
            input_ids = batch[0].to(device)
            att_masks = batch[1].to(device)
            tgt = batch[2].to(device)

            masked_logits, _, _ = model(input_ids, att_masks)

            pred = torch.argmax(masked_logits, dim=-1)

            loss = F.cross_entropy(masked_logits, tgt)
            loss_total += loss

            preds.extend(pred.cpu().tolist())
            labels.extend(tgt.cpu().detach().tolist())

    model.train()
    macro_F1 = f1_score(labels, preds, average='macro')
    return loss_total / len(data_loader), macro_F1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_domain',
                        choices=['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst',
                                 'ch', 'f', 'gw', 'os', 's', 'fiction', 'government',
                                 'state', 'telephone', 'travel', 'sick', 'snli'],
                        default='book')
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--cuda', type=str, default=0)
    parser.add_argument('--model', type=str, default='roberta-base')
    args = parser.parse_args()

    device = torch.device('cuda:%s' % args.cuda if torch.cuda.is_available() else 'cpu')

    seed = args.seed
    set_seed(seed)

    sa_domains = ['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst']
    rumour_domains = ['ch', 'f', 'gw', 'os', 's']
    nli_domains = ['fiction', 'government', 'state', 'telephone', 'travel', 'sick', 'snli']

    target_domain = args.target_domain

    if target_domain in nli_domains:
        template = '\'[sentence1]\'? <mask>, \'[sentence2]\''
    else:
        template = 'It was <mask>. '

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

    bert_model_path = '/sdc1/liqizhi/huggingface/%s' % args.model
    tokenizer = RobertaTokenizer.from_pretrained(bert_model_path)

    amazon_dir_path = 'datasets/amazon'
    imdb_dir_path = 'datasets/imdb'
    sst_dir_path = 'datasets/sst'
    pheme_dir_path = 'datasets/PHEME'
    mnli_dir_path = 'datasets/NLI/MNLI'
    sick_dir_path = 'datasets/NLI/SICK'
    snli_dir_path = 'datasets/NLI/SNLI'

    if target_domain in sa_domains:
        task_name = 'sa'
        if target_domain == 'sst' or target_domain == 'imdb':
            num_domains = len(os.listdir(amazon_dir_path)) + 1
            if target_domain == 'sst':
                max_length = 128
            else:
                max_length = 196
        else:
            num_domains = len(os.listdir(amazon_dir_path))
            max_length = 128
    elif target_domain in rumour_domains:
        task_name = 'rumour'
        max_length = 64
        num_domains = len(os.listdir(pheme_dir_path))
    elif target_domain in nli_domains:
        task_name = 'nli'
        single_sentence_max_length = 48
        max_length = 128
        if target_domain == 'sick' or target_domain == 'snli':
            num_domains = len(os.listdir(mnli_dir_path)) + 1
        else:
            num_domains = len(os.listdir(mnli_dir_path))

    batch_size = 16
    lr = 1e-5

    num_epochs = 30

    save_dir = 'parameters/PDA_{}.bin'.format(target_domain)

    verbalizer_ids, index2ids = obtain_verbalizer_ids(verbalizer, tokenizer)

    if target_domain in sa_domains:
        if target_domain == 'sst' or target_domain == 'imdb':
            train_datasets = load_amazon_dataset(amazon_dir_path, target_domain, template, tokenizer, max_length)

            if target_domain == 'imdb':
                val_dataset, test_dataset = load_imdb_dataset(imdb_dir_path, tokenizer, max_length, template)
            else:
                val_dataset, test_dataset = load_sst_dataset(sst_dir_path, tokenizer, max_length, template)

            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
        else:
            train_datasets, test_dataset = load_amazon_dataset(amazon_dir_path, target_domain, template, tokenizer, max_length)
    elif target_domain in rumour_domains:
        train_datasets, test_dataset = load_pheme_dataset(pheme_dir_path, target_domain, tokenizer, max_length, template)
    elif target_domain in nli_domains:
        if target_domain == 'sick':
            train_datasets = load_mnli_datasets(mnli_dir_path, target_domain, tokenizer,
                                                single_sentence_max_length, max_length, template)
            val_dataset, test_dataset = load_snli_sick_datasets(sick_dir_path, tokenizer,
                                                                single_sentence_max_length,
                                                                max_length, template)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
        elif target_domain == 'snli':
            train_datasets = load_mnli_datasets(mnli_dir_path, target_domain, tokenizer,
                                                single_sentence_max_length, max_length, template)
            val_dataset, test_dataset = load_snli_sick_datasets(snli_dir_path, tokenizer,
                                                                single_sentence_max_length,
                                                                max_length, template)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
        else:
            train_datasets, test_dataset = load_mnli_datasets(mnli_dir_path, target_domain, tokenizer,
                                                              single_sentence_max_length, max_length, template)

    train_dataloader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
                        for train_dataset in train_datasets]
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    mlm_model = MaksedLanguageModel(bert_model_path, tokenizer, verbalizer_ids).to(device)
    if 'base' in args.model:
        domain_adv = DomainDiscriminators(num_domains, lr, in_size=768).to(device)
    else:
        domain_adv = DomainDiscriminators(num_domains, lr, in_size=1024).to(device)

    domain_kl = DomainKL(num_domains).to(device)

    ce_criteria = nn.CrossEntropyLoss()

    model_parameters = count_parameters(mlm_model)
    model_parameters += count_parameters(domain_adv)
    print('Number of parameters: {:.4f}M'.format(model_parameters))

    param_list = [
        {'params': mlm_model.parameters()},
        {'params': domain_adv.parameters()},
    ]

    optimizer = AdamW(
        param_list,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1.0e-8
    )

    num_batches = len(train_dataloader[0])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=num_epochs * num_batches,
    )

    best_macro_F1 = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_train_steps = 0
        pbar = tqdm(zip(*train_dataloader), desc='Seed {}, Target domain {}, Training epoch {}'.format(
            seed, target_domain, epoch))

        start_time = time.time()

        for step, batches in enumerate(pbar):
            epoch_time = time.time()
            mlm_model.train()

            logits_list, labels = [], []
            logits_pos_list, logits_neg_list = [], []
            hidden_pos_list, hidden_neg_list = [], []

            for idx, batch in enumerate(batches):
                input_ids = batch[0].to(device)
                att_masks = batch[1].to(device)
                tgt = batch[2].to(device)

                masked_logits, masked_hidden_states, _ = mlm_model(input_ids, att_masks)

                logits_list.append(masked_logits)

                logits_pos_list.append(masked_logits[torch.where(tgt > 0)])
                hidden_pos_list.append(masked_hidden_states[torch.where(tgt > 0)])
                logits_neg_list.append(masked_logits[torch.where(tgt == 0)])
                hidden_neg_list.append(masked_hidden_states[torch.where(tgt == 0)])

                labels.append(tgt)

            logits = torch.cat(logits_list, dim=0)
            labels = torch.cat(labels, dim=0)

            loss_ce = ce_criteria(logits, labels)
            loss = loss_ce

            kl_pos = domain_kl(logits_pos_list)
            kl_neg = domain_kl(logits_neg_list)
            loss_kl = (kl_pos + kl_neg) * 0.5
            loss += loss_kl * 0.1

            loss_domain_pos, _, _ = domain_adv(hidden_pos_list)
            loss_domain_neg, _, _ = domain_adv(hidden_neg_list)
            loss_domain = (loss_domain_pos + loss_domain_neg) * 0.5
            loss = loss + loss_domain * 0.1

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_train_steps += 1

        _, macro_F1 = evaluate(mlm_model, test_dataloader, target_domain)

        if macro_F1 > best_macro_F1:
            best_macro_F1 = macro_F1
            torch.save(mlm_model.state_dict(), save_dir)
        print("Epoch {}, F1: {:.4f}, ".format(epoch, macro_F1))

    best_f1 = best_macro_F1
    print('best macro-F1: {:.4f}'.format(best_macro_F1))

    if not os.path.exists('results/scores/%s/PDA/%s' % (task_name, args.model)):
        os.makedirs('results/scores/%s/PDA/%s' % (task_name, args.model))

    with open('results/scores/%s/PDA/%s/%s.txt' % (task_name, args.model, target_domain), 'a') as f:
        f.write('f1: %.4f\n' % best_f1)
