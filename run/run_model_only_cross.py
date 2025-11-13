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
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

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
    def __init__(self, data, targets, tokenizer, max_length, domains=None):
        self.data = data
        self.targets = targets
        self.domains = domains
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        target = int(self.targets[index])

        if self.domains is not None:
            domain = self.domains[index]

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

        if self.domains is None:
            return input_ids, attention_mask, target
        else:
            return input_ids, attention_mask, target, domain


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
    d_verbalizer = {}

    domains = os.listdir(dir_path)
    domain_mapping = {
        'ch': 'charliehebdo',
        'f': 'ferguson',
        'gw': 'germanwings',
        'os': 'ottawashooting',
        's': 'sydneysiege',
    }
    domain_verbalizer_mapping = {
        'charliehebdo': 'magazine',
        'ferguson': 'human',
        'germanwings': 'company',
        'ottawashooting': 'shoot',
        'sydneysiege': 'siege',
    }

    i = 0  # domain 的标签
    for domain in domains:
        if domain != '{}.csv'.format(domain_mapping[target_domain]):
            texts, labels, train_domain_labels = read_pheme_csv(os.path.join(dir_path, domain), template, i)
            train_dataset = CustomDataset(texts, labels, tokenizer, max_length, train_domain_labels)
            train_datasets.append(train_dataset)
            d_verbalizer[domain_verbalizer_mapping[domain.split('.')[0]]] = i
            i += 1

    texts, labels = read_pheme_csv(os.path.join(dir_path, '{}.csv'.format(domain_mapping[target_domain])), template)
    test_dataset = CustomDataset(texts, labels, tokenizer, max_length)

    return train_datasets, test_dataset, d_verbalizer


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


def load_amazon_dataset(dir_path, target_domain, tokenizer, max_length, template):
    train_texts, train_labels, train_domains, train_datasets, train_domain_labels = [], [], [], [], []
    test_texts, test_labels = [], []
    d_verbalizer = {}

    domains = os.listdir(dir_path)

    i = 0  # domain 的标签
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
                    train_domain_labels.append(i)
            d_verbalizer[domain] = i
            i += 1

    for num_samples in train_domains:
        train_dataset = CustomDataset(train_texts[:num_samples], train_labels[:num_samples],
                                      tokenizer, max_length, train_domain_labels[:num_samples])
        del train_texts[:num_samples]
        del train_labels[:num_samples]
        del train_domain_labels[:num_samples]
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

        return train_datasets, test_dataset, d_verbalizer
    else:
        return train_datasets, d_verbalizer


def read_nli_csv(dir_path, template, tokenizer, single_sentence_max_length, i=None):
    df = pd.read_csv(dir_path, header=0)
    sentence1 = df['sentence1'].tolist()
    sentence2 = df['sentence2'].tolist()
    labels = df['label'].tolist()
    domain_labels = []

    texts = []
    for index in range(len(sentence1)):
        # 如果文本过长, 将文本截断, 同时去除收尾[CLS] [SEP]
        sentence1_truncation = tokenizer.decode(tokenizer.encode(sentence1[index],
                                                                 truncation=True,
                                                                 max_length=single_sentence_max_length)[1: -1])
        sentence2_truncation = tokenizer.decode(tokenizer.encode(sentence2[index],
                                                                 truncation=True,
                                                                 max_length=single_sentence_max_length)[1: -1])

        text = template.replace("[sentence1]", '\'{}\''.format(sentence1_truncation)).replace("[sentence2]",
                                                                                              '\'{}\''.format(
                                                                                                  sentence2_truncation))
        texts.append(text)
        if i is not None:
            domain_labels.append(i)

    if i is not None:
        return texts, labels, domain_labels
    else:
        return texts, labels


def load_mnli_datasets(dir_path, target_domain, tokenizer, single_sentence_max_length, max_length, template):
    train_datasets, train_domain_labels = [], []
    domains = os.listdir(dir_path)
    d_verbalizer = {}

    i = 0  # domain 的标签
    for domain in domains:
        if domain != target_domain:
            texts, labels, domain_labels = read_nli_csv(os.path.join(dir_path, domain, 'train.csv'),
                                                        template, tokenizer, single_sentence_max_length, i)
            train_dataset = CustomDataset(texts, labels, tokenizer, max_length, domain_labels)
            train_datasets.append(train_dataset)

            d_verbalizer[domain] = i
            i += 1

    if target_domain in domains:
        test_texts, test_labels = read_nli_csv(os.path.join(dir_path, target_domain, 'test.csv'),
                                               template, tokenizer, single_sentence_max_length)

        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

        return train_datasets, test_dataset, d_verbalizer
    else:
        return train_datasets, d_verbalizer


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
    # return sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'roberta' not in name) / 1e6  # 将参数量转换为百万（M）
    return num_parameters / 1e6  # 将参数量转换为百万（M）


def obtain_verbalizer_ids(verbalizer, tokenizer):
    """
    将 verbalizer 中的词语转成 Embedding layer 的 id
    :param verbalizer: dict
    :param tokenizer: Object
    :return verbalizer_ids: list
            verbalizer 中所有词语在 Embedding layer 中的 id
    :return index2ids: dict
            verbalizer_ids 中 index 与 token id 之间的映射
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

            logits, _, _, _, _ = model(input_ids, att_masks)

            pred = torch.argmax(logits, dim=-1)
            loss = F.cross_entropy(logits, tgt)
            loss_total += loss
            preds.extend(pred.cpu().detach().tolist())

    model.train()
    macro_F1 = f1_score(labels, preds, average='macro')

    # accuracy = accuracy_score(labels, preds)
    # precision = precision_score(labels, preds)
    # recall = recall_score(labels, preds)
    # f1 = f1_score(labels, preds)
    # return accuracy, precision, recall, f1
    return loss_total / len(data_loader), macro_F1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scl', action='store_true', help='using sentence contrastive learning')
    parser.add_argument('--md_adv', action='store_true', help='using masked domain adversarial training')
    parser.add_argument('--target_domain',
                        choices=['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst',
                                 'ch', 'f', 'gw', 'os', 's', 'fiction', 'government',
                                 'slate', 'telephone', 'travel', 'sick', 'snli'],
                        default='book')
    # parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--cuda', type=int, default=2)
    parser.add_argument('--save_losses', action='store_true', help='saving losses curve or not')
    parser.add_argument('--cov', action='store_true')
    # parser.add_argument('--hyper', nargs='+', type=float)
    args = parser.parse_args()

    print(args)

    if not args.cov:
        alpha = args.hyper[0]
        beta = args.hyper[1]
        sigma = args.hyper[2]

    if args.save_losses:
        losses_curve_path = 'results/losses_curve_{}.txt'.format(args.target_domain)
        weight_curve_path = 'results/weight_curve_{}.txt'.format(args.target_domain)
        if os.path.exists(losses_curve_path):
            os.remove(losses_curve_path)

        if os.path.exists(weight_curve_path):
            os.remove(weight_curve_path)

    device = torch.device('cuda:%s' % args.cuda if torch.cuda.is_available() else 'cpu')

    # seed = args.seed
    for _ in range(0, 1000):
        seed = random.choice(range(0, 9999))
        set_seed(seed)

        sa_domains = ['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst']
        rumor_domains = ['ch', 'f', 'gw', 'os', 's']
        nli_domains = ['fiction', 'government', 'slate', 'telephone', 'travel', 'sick', 'snli']

        model_path = '/sdc1/liqizhi/huggingface/roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        target_domain = args.target_domain

        # BERT template
        # template = 'It was [MASK]. '
        # RoBERTa template
        if target_domain in nli_domains:
            # s_template = ('Given that the sentence [sentence1] is true, the relationship between [sentence1] and ['
            #               'sentence2] is <mask>.')
            s_template = '\'[sentence1]\'? <mask>, \'[sentence2]\''
            d_template = 'A pair of sentences from the domain of <mask>.'
        else:
            s_template = 'It was <mask>. '
            d_template = 'A sentence from the domain of <mask>.'
        template = d_template + tokenizer.sep_token + s_template

        # model_path = '../pretrained_parameters/bert-base-uncased'
        amazon_dir_path = 'datasets/amazon'
        imdb_dir_path = 'datasets/imdb'
        sst_dir_path = 'datasets/sst'
        pheme_dir_path = 'datasets/PHEME'
        mnli_dir_path = 'datasets/NLI/MNLI'
        sick_dir_path = 'datasets/NLI/SICK'
        snli_dir_path = 'datasets/NLI/SNLI'

        # num_domains = len(os.listdir(dir_path))
        save_dir = 'parameters/model-scl {}-md_adv {}-target {}.bin'.format(args.scl, args.md_adv, target_domain)

        if target_domain in sa_domains:
            s_verbalizer = {
                'good': 1,
                'bad': 0
            }
        elif target_domain in rumor_domains:
            s_verbalizer = {
                'rumour': 1,
                'truth': 0
            }
        elif target_domain in nli_domains:
            s_verbalizer = {
                'uncertain': 0,
                'no': 1,
                'yes': 2,
            }

        if target_domain == 'imdb':
            max_length = 196
        elif target_domain in sa_domains:
            max_length = 128
        elif target_domain in rumor_domains:
            max_length = 64
        elif target_domain in nli_domains:
            single_sentence_max_length = 48
            max_length = 128

        batch_size = 16

        weight_decay = 1e-2
        lr = 1e-5  # 0.000015
        num_epochs = 15

        if target_domain in sa_domains:
            if target_domain != 'sst' and target_domain != 'imdb':
                train_datasets, test_dataset, d_verbalizer = load_amazon_dataset(amazon_dir_path, target_domain,
                                                                                 tokenizer,
                                                                                 max_length, template)
            else:
                train_datasets, d_verbalizer = load_amazon_dataset(amazon_dir_path, target_domain, tokenizer,
                                                                   max_length, template)

                if args.target_domain == 'imdb':
                    val_dataset, test_dataset = load_imdb_dataset(imdb_dir_path, tokenizer, max_length, template)
                else:
                    val_dataset, test_dataset = load_sst_dataset(sst_dir_path, tokenizer, max_length, template)

                # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
        elif target_domain in rumor_domains:
            train_datasets, test_dataset, d_verbalizer = load_pheme_dataset(pheme_dir_path, target_domain, tokenizer,
                                                                            max_length, template)
        elif target_domain in nli_domains:
            if target_domain == 'sick':
                train_datasets, d_verbalizer = load_mnli_datasets(mnli_dir_path, target_domain, tokenizer,
                                                                  single_sentence_max_length,
                                                                  max_length, template)
                val_dataset, test_dataset = load_snli_sick_datasets(sick_dir_path, tokenizer,
                                                                    single_sentence_max_length,
                                                                    max_length, template)
                # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
            elif target_domain == 'snli':
                train_datasets, d_verbalizer = load_mnli_datasets(mnli_dir_path, target_domain, tokenizer,
                                                                  single_sentence_max_length,
                                                                  max_length, template)
                val_dataset, test_dataset = load_snli_sick_datasets(snli_dir_path, tokenizer,
                                                                    single_sentence_max_length,
                                                                    max_length, template)
                # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
            else:
                train_datasets, test_dataset, d_verbalizer = load_mnli_datasets(mnli_dir_path, target_domain, tokenizer,
                                                                                single_sentence_max_length,
                                                                                max_length, template)

        train_dataloader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
                            for train_dataset in train_datasets]
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

        s_verbalizer_ids, s_index2ids = obtain_verbalizer_ids(s_verbalizer, tokenizer)
        d_verbalizer_ids, d_index2ids = obtain_verbalizer_ids(d_verbalizer, tokenizer)

        model = KW.Model(model_path, tokenizer, s_verbalizer_ids, d_verbalizer_ids).to(device)
        if target_domain in nli_domains:
            so = KW.SentenceOrthogonality(768, 256, device, num_classes=3).to(device)
        else:
            so = KW.SentenceOrthogonality(768, 256, device).to(device)

        criteria = nn.CrossEntropyLoss()

        model_parameters = count_parameters(model)
        model_parameters += count_parameters(so)
        print('Number of parameters: {:.4f}M'.format(model_parameters))

        mean_sort = 'full'
        mean_decay_param = 1.0
        cov_weighting_loss = KW.CoVWeightingLoss(mean_sort, mean_decay_param, device, save_losses=args.save_losses,
                                                 target_domain=args.target_domain)

        param_list = [
            {'params': model.parameters()},
            {'params': so.parameters()},
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

        # best_F1, best_acc, best_p, best_r = 0.0, 0.0, 0.0, 0.0
        best_F1 = 0.0
        best_val_loss = float('inf')
        # epoch_time = time.time()
        batch_index = 0
        epoch_index = 0
        last_improve = 0
        require_improvement = 5
        flag = False
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_train_steps = 0

            # start_time = time.time()
            # train_iters = zip(*train_dataloader)
            pbar = tqdm.tqdm(zip(*train_dataloader), desc='Seed {}, Training epoch {}'.format(seed, epoch))
            for step, batches in enumerate(pbar):
                model.train()

                # logits_list: 存储 MASK 的 logits
                # masked_hidden_states_list: 存储 MASK 的隐藏状态
                # sentence_hidden_states_list: 存储 CLS 的隐藏状态
                # labels: 存储标签
                s_logits_list, d_logits_list, cls_hidden_states_list, labels, domain_label_list = [], [], [], [], []
                pos_cls_list, neg_cls_list = [], []

                # 这里能够保证每次添加的数据都是不同域的数据,
                # 因为 train_dataloader 中遍历的就是不同域的数据
                for idx, batch in enumerate(batches):
                    input_ids = batch[0].to(device)
                    att_masks = batch[1].to(device)
                    tgt = batch[2].to(device)
                    domains = batch[3].tolist()

                    # logits: (batch_size, num_outputs), MASK 的 logits
                    # masked_hidden_states: (batch_size, hidden_size), MASK 的隐藏状态
                    # cls_hidden_states: (batch_size, hidden_size), CLS 的隐藏状态
                    s_masked_logits, d_masked_logits, masked_hidden_states, cls_hidden_states, last_hidden_states = model(
                        input_ids, att_masks)

                    cls_hidden_states_list.append(cls_hidden_states)
                    s_logits_list.append(s_masked_logits)
                    d_logits_list.append(d_masked_logits)

                    labels.append(tgt)
                    domain_label_list.append(domains)

                s_logits = torch.cat(s_logits_list, dim=0)
                d_logits = torch.cat(d_logits_list, dim=0)
                labels = torch.cat(labels, dim=0).to(device)

                if args.scl:
                    sentence_representations = torch.cat(cls_hidden_states_list, dim=0)
                    s_contrastive_loss = so(sentence_representations, labels)

                if args.md_adv:
                    shuffled_list = domain_label_list[:]
                    while any(shuffled_list[i] == domain_label_list[i] for i in range(len(domain_label_list))):
                        random.shuffle(shuffled_list)

                    try:
                        domain_labels = torch.tensor(shuffled_list).view(-1).to(device)
                    except ValueError:
                        # 解决一个batch中大小不一的问题
                        domain_labels = torch.tensor(list(itertools.chain(*shuffled_list))).to(device)

                    md_loss = criteria(d_logits, domain_labels)

                ce_loss = criteria(s_logits, labels)

                if not args.md_adv and not args.scl:
                    loss = ce_loss
                elif args.scl and args.md_adv:
                    if args.save_losses:
                        loss = cov_weighting_loss([ce_loss, md_loss, s_contrastive_loss],
                                                  losses_names=['ce', 'md_adv', 'pcl'], iteration=batch_index)
                        batch_index += 1
                    else:
                        if args.cov:
                            loss = cov_weighting_loss([ce_loss, md_loss, s_contrastive_loss])
                        else:
                            loss = alpha * ce_loss + beta * md_loss + sigma * s_contrastive_loss
                elif args.md_adv and not args.scl:
                    loss = cov_weighting_loss([ce_loss, md_loss])
                elif args.scl and not args.md_adv:
                    loss = cov_weighting_loss([ce_loss, s_contrastive_loss])

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                # scheduler.step()

                total_loss += loss.item()
                num_train_steps += 1

            _, f1 = evaluate(model, test_dataloader, target_domain)

            if f1 > best_F1:
                best_F1 = f1
                last_improve = epoch_index
                torch.save(model.state_dict(), save_dir)

            print("Epoch {}, F1: {:.4f}, ".format(epoch, f1))

        print('Best F1: {:.4f}'.format(best_F1))

        if not os.path.exists('results/scores/GenPromptCL/'):
            os.makedirs('results/scores/GenPromptCL/')

        with open(f'results/scores/GenPromptCL/{target_domain}.txt', 'a') as f:
            f.write('seed: %d, \t f1: %.4f\n' % (seed, best_F1))

        os.remove(save_dir)
