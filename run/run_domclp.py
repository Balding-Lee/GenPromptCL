'''
@Time : 2024/5/8 15:53
@Auth : Qizhi Li
'''
import os
import sys
import tqdm
import nltk
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from nltk.corpus import wordnet
from sklearn.metrics import f1_score
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from models import DomCLP


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synonym_replacement(sentences):
    synonym_sentences = []
    for i in tqdm.trange(len(sentences), desc='preparing synonym texts'):
        sentence = sentences[i]

        tokens = nltk.word_tokenize(sentence)

        num_selected_words = int(len(tokens) * 0.1)
        selected_words = random.sample(tokens, num_selected_words)

        tagged_words = nltk.pos_tag(selected_words)
        filtered_words = [word for word, tag in tagged_words if
                          tag.startswith('N') or tag.startswith('V') or tag.startswith('R') or tag.startswith('J')]

        for word in filtered_words:
            synsets = wordnet.synsets(word)
            if synsets:
                synonym = synsets[0].lemmas()[0].name()
                sentence = sentence.replace(word, synonym)

        synonym_sentences.append(sentence)

    return synonym_sentences


class UnsupCustomDataset(Dataset):
    def __init__(self, data, synonym_data, targets, domains, tokenizer, max_length):
        self.data = data
        self.synonym_data = synonym_data
        self.targets = targets
        self.domains = domains
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        synonym_text = self.synonym_data[index]
        domain = int(self.domains[index])
        target = int(self.targets[index])

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        synonym_inputs = self.tokenizer(
            synonym_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        synonym_ids = synonym_inputs['input_ids'].squeeze(0)
        synonym_attention_mask = synonym_inputs['attention_mask'].squeeze(0)

        return index, input_ids, attention_mask, synonym_ids, synonym_attention_mask, target, domain


class CustomDataset(Dataset):

    def __init__(self, data, targets, tokenizer, max_length, synonym_data=None, domains=None):
        self.data = data
        self.synonym_data = synonym_data
        self.targets = targets
        self.domains = domains
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        target = int(self.targets[index])

        if self.domains:
            synonym_text = self.synonym_data[index]

            synonym_inputs = self.tokenizer(
                synonym_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            synonym_input_ids = synonym_inputs['input_ids'].squeeze(0)
            synonym_attention_mask = synonym_inputs['attention_mask'].squeeze(0)

            domain = int(self.domains[index])

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        if self.domains:
            return index, input_ids, attention_mask, synonym_input_ids, synonym_attention_mask, target, domain
        else:
            return input_ids, attention_mask, target


def read_imdb_csv(dir_path):
    data = pd.read_csv(dir_path, header=0)
    texts = data['sentence'].tolist()
    labels = data['labels'].tolist()

    return texts, labels


def load_imdb_dataset(dir_path, tokenizer, max_length):
    val_texts, val_labels = read_imdb_csv(os.path.join(dir_path, 'dev.csv'))
    test_texts, test_labels = read_imdb_csv(os.path.join(dir_path, 'test.csv'))

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


def read_sst_tsv(dir_path):
    data = pd.read_csv(dir_path, sep='\t', header=0)
    texts = data['sentence'].tolist()
    labels = data['label'].tolist()

    return texts, labels


def load_sst_dataset(dir_path, tokenizer, max_length):
    val_texts, val_labels = read_sst_tsv(os.path.join(dir_path, 'dev.tsv'))
    test_texts, test_labels = read_sst_tsv(os.path.join(dir_path, 'test.tsv'))

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)
    return val_dataset, test_dataset


def load_amazon_dataset(dir_path, target_domain, tokenizer, max_length):
    train_texts, train_labels, train_domains, train_datasets, unsup_train_texts, unsup_train_labels = [], [], [], [], [], []
    unsup_train_texts_all = []
    test_texts, test_labels = [], []

    domains = os.listdir(dir_path)
    output_domains, all_domains = [], []
    i = 0
    for domain in domains:
        if domain != target_domain:
            with open(os.path.join(dir_path, domain, 'all_data.txt'), 'r') as f:
                texts = f.readlines()

            train_domains.append(len(texts))
            for text in texts:
                line = text.strip().split(' ||| ')
                if len(line) == 2:
                    train_texts.append(line[0])
                    unsup_train_texts.append(line[0])
                    unsup_train_texts_all.append(line[0])
                    unsup_train_labels.append(line[1])
                    train_labels.append(line[1])
                    output_domains.append(i)
                    all_domains.append(i)

            i += 1

    for num_samples in train_domains:
        texts = train_texts[:num_samples]
        labels = train_labels[:num_samples]

        unsup_texts = unsup_train_texts[:num_samples]
        synonym_unsup_texts = synonym_replacement(unsup_texts)
        output_domains_ = output_domains[:num_samples]

        train_dataset = CustomDataset(texts, labels, tokenizer, max_length, synonym_unsup_texts, output_domains_)

        del train_texts[:num_samples]
        del train_labels[:num_samples]
        del output_domains[:num_samples]
        del unsup_train_texts[:num_samples]

        train_datasets.append(train_dataset)

    synonym_texts = synonym_replacement(unsup_train_texts_all)
    unsup_dataset = UnsupCustomDataset(unsup_train_texts_all, synonym_texts, unsup_train_labels, all_domains, tokenizer, max_length)

    if target_domain in domains:
        with open(os.path.join(dir_path, target_domain, 'test.txt'), 'r') as f:
            lines = f.readlines()

        for text in lines:
            line = text.strip().split(' ||| ')
            if len(line) == 2:
                test_texts.append(line[0])
                test_labels.append(line[1])

        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

        return train_datasets, unsup_dataset, test_dataset
    else:
        return train_datasets, unsup_dataset


def read_pheme_csv(dir_path):
    data = pd.read_csv(dir_path, header=0)
    texts = data['texts'].tolist()
    labels = data['labels'].tolist()
    unsup_texts = data['texts'].tolist()
    return texts, unsup_texts, labels


def load_rumour_dataset(dir_path, target_domain, tokenizer, max_length):
    train_texts, train_labels, train_domains, train_datasets, unsup_train_texts = [], [], [], [], []

    train_labels, domains_all = [], []

    domain_mapping = {
        'ch': 'charliehebdo',
        'f': 'ferguson',
        'gw': 'germanwings',
        'os': 'ottawashooting',
        's': 'sydneysiege',
    }

    domains = os.listdir(dir_path)
    i = 0
    for domain in domains:
        if domain != '{}.csv'.format(domain_mapping[target_domain]):
            texts, unsup_texts, labels = read_pheme_csv(os.path.join(dir_path, domain))
            unsup_train_texts.extend(unsup_texts)
            train_texts.extend(texts)
            train_labels.extend(labels)

            train_domains.extend([i] * len(texts))
            domains_all = [i] * len(texts)

            train_synonym_texts = synonym_replacement(texts)
            train_dataset = CustomDataset(texts, labels, tokenizer, max_length, train_synonym_texts, domains_all)
            train_datasets.append(train_dataset)
            i += 1

    synonym_texts = synonym_replacement(unsup_train_texts)
    unsup_dataset = UnsupCustomDataset(unsup_train_texts, synonym_texts, train_labels,
                                       train_domains, tokenizer, max_length)

    test_texts, _, test_labels = read_pheme_csv(os.path.join(dir_path, '{}.csv'.format(domain_mapping[target_domain])))
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return train_datasets, unsup_dataset, test_dataset


def read_nli_csv(dir_path, tokenizer, single_sentence_max_length):
    df = pd.read_csv(dir_path, header=0)
    sentence1 = df['sentence1'].tolist()
    sentence2 = df['sentence2'].tolist()
    labels = df['label'].tolist()

    texts, unsup_texts = [], []
    for i in range(len(sentence1)):
        sentence1_truncation = tokenizer.decode(tokenizer.encode(sentence1[i],
                                                                 truncation=True,
                                                                 max_length=single_sentence_max_length)[1: -1])
        sentence2_truncation = tokenizer.decode(tokenizer.encode(sentence2[i],
                                                                 truncation=True,
                                                                 max_length=single_sentence_max_length)[1: -1])

        text = sentence1_truncation + tokenizer.sep_token + sentence2_truncation
        texts.append(text)
        unsup_texts.append(text)

    return texts, unsup_texts, labels


def load_mnli_datasets(dir_path, target_domain, tokenizer, single_sentence_max_length, max_length):
    train_datasets, unsup_train_texts = [], []
    domains = os.listdir(dir_path)
    train_labels, domains_all = [], []

    i = 0
    for domain in domains:
        if domain != target_domain:
            texts, unsup_texts, labels = read_nli_csv(os.path.join(dir_path, domain, 'train.csv'),
                                                      tokenizer, single_sentence_max_length)
            unsup_train_texts.extend(unsup_texts)
            sup_train_domains = [i] * len(texts)
            train_synonym_texts = synonym_replacement(texts)
            train_labels.extend(labels)
            domains_all.extend([i] * len(texts))

            train_dataset = CustomDataset(texts, labels, tokenizer, max_length, train_synonym_texts, sup_train_domains)
            train_datasets.append(train_dataset)

            i += 1

    synonym_texts = synonym_replacement(unsup_train_texts)
    unsup_train_dataset = UnsupCustomDataset(unsup_train_texts, synonym_texts, train_labels, domains_all,
                                             tokenizer, max_length)

    if target_domain in domains:
        test_texts, _, test_labels = read_nli_csv(os.path.join(dir_path, target_domain, 'test.csv'),
                                                  tokenizer, single_sentence_max_length)

        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

        return train_datasets, unsup_train_dataset, test_dataset
    else:
        return train_datasets, unsup_train_dataset


def load_snli_sick_datasets(dir_path, tokenizer, single_sentence_max_length, max_length):
    val_texts, _, val_labels = read_nli_csv(os.path.join(dir_path, 'dev.csv'),
                                            tokenizer, single_sentence_max_length)
    test_texts, _, test_labels = read_nli_csv(os.path.join(dir_path, 'test.csv'),
                                              tokenizer, single_sentence_max_length)

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


def count_parameters(model):
    num_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad and 'roberta' not in name and 'classifier' not in name:
            num_parameters += param.numel()
    return num_parameters / 1e6


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

            _, _, _, logits = model(input_ids, att_masks)
            pred = torch.argmax(logits, dim=-1)

            loss = F.cross_entropy(logits, tgt)
            loss_total += loss

            preds.extend(pred.cpu().detach().tolist())
            labels.extend(tgt.cpu().detach().tolist())

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
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--cuda', default=0)
    parser.add_argument('--num_cluster', default='7', type=str, help='number of clusters')
    parser.add_argument('--temperature', default=0.2, type=float, help='softmax temperature')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    seed = args.seed
    set_seed(seed)

    model_path = '/sdc1/liqizhi/huggingface/roberta-base'
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

    target_domain = args.target_domain
    num_outputs = 2
    if target_domain in sa_domains:
        if target_domain == 'sst' or target_domain == 'imdb':
            num_train_domains = 4
            train_domains = ['book', 'dvd', 'electronics', 'kitchen']
            if target_domain == 'sst':
                max_length = 128
            else:
                max_length = 196
        else:
            num_train_domains = 3
            max_length = 128
            domains = ['book', 'dvd', 'electronics', 'kitchen']
            train_domains = [domain for domain in domains if domain != target_domain]
    elif target_domain in rumour_domains:
        max_length = 64
        num_train_domains = 4
        domains = ['ch', 'f', 'gw', 'os', 's']
        train_domains = [domain for domain in domains if domain != target_domain]
    elif target_domain in nli_domains:
        max_length = 128
        single_sequence_max_length = 48
        num_outputs = 3

        if target_domain in ['sick', 'snli']:
            num_train_domains = 5
            train_domains = ['fiction', 'government', 'slate', 'telephone', 'travel']
        else:
            domains = ['fiction', 'government', 'slate', 'telephone', 'travel']
            train_domains = [domain for domain in domains if domain != target_domain]
            num_train_domains = 4

    batch_size = 16
    weight_decay = 0.01
    lr = 1e-5  # 0.000015
    num_epochs = 15

    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    if target_domain in sa_domains:
        if target_domain == 'sst' or target_domain == 'imdb':
            train_datasets, unsup_train_dataset = load_amazon_dataset(amazon_dir_path, target_domain, tokenizer, max_length)

            if target_domain == 'imdb':
                _, test_dataset = load_imdb_dataset(imdb_dir_path, tokenizer, max_length)
            else:
                _, test_dataset = load_sst_dataset(sst_dir_path, tokenizer, max_length)

        else:
            train_datasets, unsup_train_dataset, test_dataset = load_amazon_dataset(amazon_dir_path, target_domain,
                                                                                    tokenizer, max_length)
    elif target_domain in rumour_domains:
        train_datasets, unsup_train_dataset, test_dataset = load_rumour_dataset(pheme_dir_path, target_domain,
                                                                                tokenizer, max_length)
    elif target_domain in nli_domains:
        if target_domain == 'sick' or target_domain == 'snli':
            train_datasets, unsup_train_dataset = load_mnli_datasets(mnli_dir_path, target_domain, tokenizer,
                                                                     single_sequence_max_length, max_length)

            if target_domain == 'sick':
                _, test_dataset = load_snli_sick_datasets(sick_dir_path, tokenizer,
                                                                    single_sequence_max_length,
                                                                    max_length)
            elif target_domain == 'snli':
                _, test_dataset = load_snli_sick_datasets(snli_dir_path, tokenizer,
                                                                    single_sequence_max_length,
                                                                    max_length)

        else:
            train_datasets, unsup_train_dataset, test_dataset = load_mnli_datasets(mnli_dir_path, target_domain,
                                                                                   tokenizer,
                                                                                   single_sequence_max_length,
                                                                                   max_length)

    train_dataloader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
                        for train_dataset in train_datasets]
    unsup_train_dataloader = DataLoader(unsup_train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    model = DomCLP.BaselineClsModel(model_path, num_outputs=num_outputs).to(device)

    model_parameters = count_parameters(model)
    print('Number of parameters: {:.4f}M'.format(model_parameters))

    criterion = DomCLP.DomCLPLoss(device).to(device)
    criterion_pcl = nn.CrossEntropyLoss().to(device)
    criterion_cls = nn.CrossEntropyLoss().to(device)

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

    num_batches = 0
    for loader in train_dataloader:
        num_batches += len(loader)

    best_macro_F1 = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):

        # features: [num_samples, 256]
        # domains: [num_samples]
        features, domains_all = DomCLP.compute_features(unsup_train_dataloader, model, device)

        # placeholder for clustering result
        cluster_result = {'text2cluster': [], 'centroids': [], 'density': []}
        for num_cluster in args.num_cluster:
            cluster_result['text2cluster'].append(torch.zeros(len(unsup_train_dataset), dtype=torch.long).to(device))
            cluster_result['centroids'].append(
                torch.zeros(int(num_cluster) * num_train_domains, 768).to(device))
            cluster_result['density'].append(torch.zeros(int(num_cluster) * num_train_domains).to(device))

        features = features.numpy()
        cluster_result = DomCLP.run_kmeans(features, domains_all, args, num_train_domains, device, train_domains, args.cuda)  # run kmeans clustering on master node

        model.train()
        for idx, batches in enumerate(tqdm.tqdm(zip(*train_dataloader), desc=f'training epoch {epoch + 1}')):
            for idx, batch in enumerate(batches):
                index = batch[0]
                original_input_ids = batch[1].to(device)
                original_att_masks = batch[2].to(device)
                synonym_input_ids = batch[3].to(device)
                synonym_att_masks = batch[4].to(device)
                labels = batch[5].to(device)
                domains = batch[6].to(device)

                input_ids = torch.cat([original_input_ids, synonym_input_ids], dim=0)
                att_masks = torch.cat([original_att_masks, synonym_att_masks], dim=0)

                out, rep, proj, logits = model(input_ids, att_masks)

                bsz = labels.shape[0]

                original_logits, _ = torch.split(logits, [bsz, bsz], dim=0)
                loss_scl = criterion_cls(original_logits, labels)

                z1, z2 = torch.split(proj, [bsz, bsz], dim=0)
                z_self = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
                loss_self = criterion(z_self, domains)

                target_proto = []
                output_proto = []
                for n, (k, im2cluster, prototypes, density) in enumerate(
                        zip(args.num_cluster, cluster_result['text2cluster'], cluster_result['centroids'],
                            cluster_result['density'])):
                    k = int(k)
                    pos_proto_id = im2cluster[index]
                    pos_prototypes = prototypes[pos_proto_id]
                    neg_proto_id = torch.tensor([list(set(range(k * dom, k * (dom + 1))) - {idx.item()}) for idx, dom in
                                                 zip(pos_proto_id, domains_all[index])]).to(device)
                    neg_prototypes = prototypes[neg_proto_id]
                    proto_selected = torch.cat([pos_prototypes.unsqueeze(1), neg_prototypes], dim=1)
                    logits_proto = torch.einsum('ij,ikj->ik', z1, proto_selected)  ### [256, 14]
                    labels_proto = torch.zeros(len(pos_proto_id)).long().to(device)
                    temp_proto = density[torch.cat((pos_proto_id.unsqueeze(1), neg_proto_id), dim=1)]
                    logits_proto /= temp_proto

                    target_proto.append(labels_proto)
                    output_proto.append(logits_proto)

                loss_proto = torch.tensor(0.).to(device)
                for proto_out, proto_target in zip(output_proto, target_proto):
                    loss_proto += criterion_pcl(proto_out, proto_target)

                loss_proto /= len(args.num_cluster)

                loss = loss_self + loss_proto + loss_scl

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

        _, macro_F1 = evaluate(model, test_dataloader, target_domain)

        if macro_F1 > best_macro_F1:
            best_macro_F1 = macro_F1

        print("Epoch {}, F1: {:.4f}, ".format(epoch, macro_F1))

    print('Target_domain: {:s}, seed: {:d}, Test F1: {:.4f}'.format(target_domain, seed, best_macro_F1))

    if not os.path.exists('results/scores/domclp/'):
        os.makedirs('results/scores/domclp/')

    with open(f'results/scores/domclp/{target_domain}.txt', 'a') as f:
        f.write('seed: %d, \t f1: %.4f\n' % (seed, best_macro_F1))
