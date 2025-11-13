'''
@Time : 2024/5/14 10:23
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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader, TensorDataset

sys.path.append('..')
import utils
from models import GenPromptCL, PDA, eagle


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
    def __init__(self, data, labels, tokenizer, max_length):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        label = int(self.labels[index])

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

        return input_ids, attention_mask, label


def load_amazon_dataset(dir_path, target_domain, tokenizer, max_length, template=None):
    train_texts, train_labels, train_domains, train_datasets, train_domain_labels = [], [], [], [], []
    test_texts, test_labels = [], []
    d_verbalizer = {}

    domains = os.listdir(dir_path)
    source_domain_names = []

    i = 0  # domain 的标签
    for domain in domains:
        if domain != target_domain:
            source_domain_names.append(domain)
            with open(os.path.join(dir_path, domain, 'all_data.txt'), 'r') as f:
                texts = f.readlines()

            train_domains.append(len(texts))
            for text in texts:
                line = text.strip().split(' ||| ')
                if len(line) == 2:
                    if template:
                        train_texts.append(template + line[0])
                    else:
                        train_texts.append(line[0])
                    train_labels.append(line[1])
                    train_domain_labels.append(i)
            d_verbalizer[domain] = i
            i += 1

    for num_samples in train_domains:
        train_dataset = CustomDataset(train_texts[:num_samples], train_labels[:num_samples],
                                      tokenizer, max_length)
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
                if template:
                    test_texts.append(template + line[0])
                else:
                    test_texts.append(line[0])
                test_labels.append(line[1])

        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

        return train_datasets, test_dataset, d_verbalizer, source_domain_names
    else:
        return train_datasets, d_verbalizer, source_domain_names


def read_pheme_csv(dir_path, template, i=None):
    data = pd.read_csv(dir_path, header=0)
    original_texts = data['texts'].tolist()
    labels = data['labels'].tolist()
    domain_labels = []

    texts = []
    for text in original_texts:
        if template:
            texts.append(template + text)
        else:
            texts.append(text)

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
    domain_mapping_reverse = {
        'charliehebdo': 'ch',
        'ferguson': 'f',
        'germanwings': 'germanwings',
        'ottawashooting': 'os',
        'sydneysiege': 's',
    }
    domain_verbalizer_mapping = {
        'charliehebdo': 'magazine',
        'ferguson': 'human',
        'germanwings': 'company',
        'ottawashooting': 'shoot',
        'sydneysiege': 'siege',
    }

    source_domain_names = []
    i = 0
    for domain in domains:
        if domain != '{}.csv'.format(domain_mapping[target_domain]):
            source_domain_names.append(domain_mapping_reverse[domain.split('.')[0]])
            texts, labels = read_pheme_csv(os.path.join(dir_path, domain), template)
            train_dataset = CustomDataset(texts, labels, tokenizer, max_length)
            train_datasets.append(train_dataset)
            d_verbalizer[domain_verbalizer_mapping[domain.split('.')[0]]] = i
            i += 1

    texts, labels = read_pheme_csv(os.path.join(dir_path, '{}.csv'.format(domain_mapping[target_domain])), template)
    test_dataset = CustomDataset(texts, labels, tokenizer, max_length)

    return train_datasets, test_dataset, d_verbalizer


def read_sst_tsv(dir_path, template):
    data = pd.read_csv(dir_path, sep='\t', header=0)
    original_texts = data['sentence'].tolist()
    labels = data['label'].tolist()

    texts = []
    for text in original_texts:
        if template:
            texts.append(template + text)
        else:
            texts.append(text)

    return texts, labels


def load_sst_dataset(dir_path, tokenizer, max_length, template):
    val_texts, val_labels = read_sst_tsv(os.path.join(dir_path, 'dev.tsv'), template)
    test_texts, test_labels = read_sst_tsv(os.path.join(dir_path, 'test.tsv'), template)

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


def read_imdb_csv(dir_path, template):
    data = pd.read_csv(dir_path, header=0)
    original_texts = data['sentence'].tolist()
    labels = data['labels'].tolist()

    texts = []
    for text in original_texts:
        if template:
            texts.append(template + text)
        else:
            texts.append(text)

    return texts, labels


def load_imdb_dataset(dir_path, tokenizer, max_length, template):
    val_texts, val_labels = read_imdb_csv(os.path.join(dir_path, 'dev.csv'), template)
    test_texts, test_labels = read_imdb_csv(os.path.join(dir_path, 'test.csv'), template)

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


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


def preparing_features(model, dataloader, desc, args):
    pos_features, neg_features = [], []
    features = []
    for batch in tqdm.tqdm(dataloader, desc=desc):
        input_ids = batch[0].to(device)
        att_masks = batch[1].to(device)
        labels = batch[2]

        if args.model == 'ours':
            _, _, _, feature, _ = model(input_ids, att_masks)
        elif args.model == 'PDA':
            _, _, feature = model(input_ids, att_masks)
        elif args.model == 'eagle':
            _, _, feature = model(input_ids, att_masks)
        elif args.model == 'deepcoral':
            feature = model.backbone.model(input_ids, att_masks).pooler_output

        pos_features.append(feature[labels == 1])
        neg_features.append(feature[labels == 0])
        features.append(feature)

    features = torch.cat(features, dim=0)
    pos_features = torch.cat(pos_features, dim=0)
    neg_features = torch.cat(neg_features, dim=0)
    return features, pos_features, neg_features


def create_feature_loader(source_features, target_features, batch_size, device):
    X = torch.cat((source_features, target_features), dim=0)
    y = torch.cat((torch.zeros(source_features.size(0)), torch.ones(target_features.size(0))), dim=0).to(device)

    dataset = TensorDataset(X, y)
    feature_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return feature_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['ours', 'PDA', 'eagle', 'deepcoral', 'mscl'], default='ours')
    parser.add_argument('--scl', action='store_true', help='using sentence contrastive learning')
    parser.add_argument('--md_adv', action='store_true', help='using masked domain adversarial training')
    parser.add_argument('--target_domain', default='book')
    parser.add_argument('--tsne', action='store_true', help='draw t-sne fig')
    parser.add_argument('--pad', action='store_true', help='calculate PAD')
    parser.add_argument('--mmd', action='store_true', help='calculate MMD')
    parser.add_argument('--seed', default=9)
    parser.add_argument('--cuda', default=0)
    args = parser.parse_args()

    print(args)

    seed = args.seed
    set_seed(seed)

    target_domain = args.target_domain

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    model_path = '/home/liqizhi/huggingface/roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    if args.model == 'ours':
        s_template = 'It was <mask>. '
        d_template = 'A sentence from the domain of <mask>.'
        template = d_template + tokenizer.sep_token + s_template
    elif args.model == 'PDA':
        template = 'It was <mask>. '
    else:
        template = None

    batch_size = 16
    epochs = 1

    sa_domains = ['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst']
    rumor_domains = ['ch', 'f', 'gw', 'os', 's']

    if args.target_domain == 'imdb':
        max_length = 196
    elif target_domain in rumor_domains:
        max_length = 64
    else:
        max_length = 128

    s_verbalizer = {
        'good': 1,
        'bad': 0
    }

    amazon_dir_path = 'datasets/amazon'
    imdb_dir_path = 'datasets/imdb'
    sst_dir_path = 'datasets/sst'
    pheme_dir_path = 'datasets/PHEME'

    if args.target_domain in sa_domains:
        if args.target_domain != 'sst' and args.target_domain != 'imdb':
            source_datasets, target_dataset, d_verbalizer, source_domain_names = load_amazon_dataset(amazon_dir_path,
                                                                                                     args.target_domain,
                                                                                                     tokenizer,
                                                                                                     max_length,
                                                                                                     template)
        else:
            source_datasets, d_verbalizer, source_domain_names = load_amazon_dataset(amazon_dir_path,
                                                                                     args.target_domain,
                                                                                     tokenizer,
                                                                                     max_length,
                                                                                     template)
            if args.target_domain == 'sst':
                _, target_dataset = load_sst_dataset(sst_dir_path, tokenizer, max_length, template)
            else:
                _, target_dataset = load_imdb_dataset(imdb_dir_path, tokenizer, max_length, template)
    elif args.target_domain in rumor_domains:
        source_datasets, target_dataset, d_verbalizer = load_pheme_dataset(pheme_dir_path, target_domain, tokenizer,
                                                                           max_length, template)

    source_dataloader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
                         for train_dataset in source_datasets]
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    s_verbalizer_ids, s_index2ids = obtain_verbalizer_ids(s_verbalizer, tokenizer)
    d_verbalizer_ids, d_index2ids = obtain_verbalizer_ids(d_verbalizer, tokenizer)

    if args.model == 'ours':
        model = GenPromptCL.Model(model_path, tokenizer, s_verbalizer_ids, d_verbalizer_ids).to(device)
    elif args.model == 'PDA':
        model = PDA.MaksedLanguageModel(model_path, tokenizer, s_verbalizer_ids).to(device)
    elif args.model == 'eagle':
        model = eagle.Backbone(model_path).to(device)
    elif args.model == 'deepcoral':
        model = GenPromptCL.DeepCORAL(model_path, num_classes=2).to(device)

    for param in model.parameters():
        param.requires_grad = False

    if args.model == 'ours':
        save_dir = 'parameters/model-scl {}-md_adv {}-target {}.bin'.format(args.scl, args.md_adv,
                                                                               args.target_domain)
    elif args.model == 'PDA' or args.model == 'eagle' or args.model == 'deepcoral':
        save_dir = 'parameters/{}_{}.bin'.format(args.model, args.target_domain)

    model.load_state_dict(torch.load(save_dir), strict=False)

    with torch.no_grad():
        model.eval()
        i = 1
        source_pos_features, source_neg_features = [], []
        source_features = []
        for source in source_dataloader:
            # shape: (2000, 768)
            features, pos_features, neg_features = preparing_features(model, source,
                                                                      desc='preparing the features of the {}-th domain'.format(
                                                                          i),
                                                                      args=args)
            source_pos_features.append(pos_features)
            source_neg_features.append(neg_features)
            source_features.append(features)
            i += 1

        target_features, target_pos_features, target_neg_features = preparing_features(model, target_dataloader,
                                                                                       desc='preparing the target domain features',
                                                                                       args=args)

    if args.pad:
        pADs = []
        for i in range(len(source_features)):
            print('The {}-th source features'.format(i))
            pAD = utils.calculate(source_features[i], target_features, device)
            pADs.append(pAD)

        print('seed %d, pAD: %.4f' % (seed, (sum(pADs) / len(pADs))))

        with open('results/pad_{}_ours.txt'.format(args.target_domain), 'a') as f:
            text = 'seed: %d\tpAD: %.4f\n' % (seed, (sum(pADs) / len(pADs)))
            f.write(text)

    if args.mmd:
        mmds = []
        for i in range(len(source_features)):
            mmd = utils.compute_mmd(target_features, source_features[i], sigma=1.0)
            mmds.append(mmd)
            print('The {}-th domain: {:.4f}'.format(i, mmd))

        print('mean MMD: %.4f' % (sum(mmds) / len(mmds)))

    # ======================= t-SNE =======================
    if args.tsne:
        if args.target_domain == 'sst' or args.target_domain == 'imdb':
            domain_names = [args.target_domain]
            domain_names.extend(source_domain_names)
        else:
            domain_names = source_domain_names
            domain_names.append(args.target_domain)
        domain_name_id_mapping, domain_id_name_mapping = {}, {}
        i = 0
        for domain_name in domain_names:
            domain_name_id_mapping[domain_name] = i
            domain_id_name_mapping[i] = domain_name
            i += 1

        label2name = {
            0: 'negative',
            1: 'positive'
        }

        if args.target_domain == 'sst' or args.target_domain == 'imdb':
            pos_features = [target_pos_features]
            pos_features.extend(source_pos_features)
        else:
            pos_features = source_pos_features
            pos_features.append(target_pos_features)
        pos_features_np = np.vstack([tensor.cpu().numpy() for tensor in pos_features])
        pos_labels = np.array([[i, 1] for i, features in enumerate(pos_features) for _ in range(len(features))])

        if args.target_domain == 'sst' or args.target_domain == 'imdb':
            neg_features = [target_neg_features]
            neg_features.extend(source_neg_features)
        else:
            neg_features = source_neg_features
            neg_features.append(target_neg_features)
        neg_features_np = np.vstack([tensor.cpu().numpy() for tensor in neg_features])
        neg_labels = np.array([[i, 0] for i, features in enumerate(neg_features) for _ in range(len(features))])

        tsne = TSNE(n_components=2,
                    random_state=seed)
        pos_data_2d = tsne.fit_transform(pos_features_np)
        neg_data_2d = tsne.fit_transform(neg_features_np)

        plt.figure(figsize=(10, 10))

        if args.target_domain == 'sst' or args.target_domain == 'imdb':
            colors = ['#F48C8C', '#F3F1AD', '#EAFFD0', '#95E1D3', '#F4D8A5']
            edgecolors = ['#f38181', '#FCE38A', '#C0F0D2', '#87CDC0', '#F4BF9D']
        else:
            colors = ['#F48C8C', '#F3F1AD', '#EAFFD0', '#95E1D3']
            edgecolors = ['#f38181', '#FCE38A', '#C0F0D2', '#87CDC0']

        for domain_id in domain_id_name_mapping:
            idx = (pos_labels[:, 0] == domain_id)
            plt.scatter(pos_data_2d[idx, 0], pos_data_2d[idx, 1], c=colors[domain_id], s=30,
                        edgecolor=edgecolors[domain_id], label=f'Domain: {domain_id_name_mapping[domain_id]}')

        plt.xticks([])
        plt.yticks([])

        plt.legend(fontsize=18, framealpha=0.5)
        if args.model == 'ours':
            plt.savefig('results/visualize/t-sne model {} target {}-pos-scl {}-md_adv {}.pdf'.format(args.model,
                                                                                                        args.target_domain,
                                                                                                        args.scl,
                                                                                                        args.md_adv))
        else:
            plt.savefig('results/visualize/t-sne model {} target {}-pos.pdf'.format(args.model, args.target_domain))

        plt.cla()

        for domain_id in domain_id_name_mapping:
            idx = (neg_labels[:, 0] == domain_id)
            plt.scatter(neg_data_2d[idx, 0], neg_data_2d[idx, 1], c=colors[domain_id], s=30,
                        edgecolor=edgecolors[domain_id], label=f'Domain: {domain_id_name_mapping[domain_id]}')

        plt.xticks([])
        plt.yticks([])

        plt.legend(fontsize=18, framealpha=0.5)
        if args.model == 'ours':
            plt.savefig('results/visualize/t-sne model {} target {}-neg-scl {}-md_adv {}.pdf'.format(args.model,
                                                                                                        args.target_domain,
                                                                                                        args.scl,
                                                                                                        args.md_adv))
        else:
            plt.savefig('results/visualize/t-sne model {} target {}-neg.pdf'.format(args.model, args.target_domain))
