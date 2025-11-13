'''
@Time : 2024/4/2 10:28
@Auth : Qizhi Li
'''
import os
import nltk
import tqdm
import random
import pandas as pd
from nltk.corpus import wordnet
from torch.utils.data import Dataset


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


class CustomDatasetAUG(Dataset):
    def __init__(self, data, synonym_data, targets, tokenizer, max_length):
        self.data = data
        self.synonym_data = synonym_data
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        synonym_text = self.synonym_data[index]
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

        return input_ids, attention_mask, synonym_ids, synonym_attention_mask, target


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


def read_pheme_csv_aug(dir_path):
    data = pd.read_csv(dir_path, header=0)
    texts = data['texts'].tolist()
    synonym_texts = synonym_replacement(texts)
    labels = data['labels'].tolist()

    return texts, synonym_texts, labels


def load_pheme_dataset_aug(dir_path, target_domain, tokenizer, max_length):
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
            texts, synonym_texts, labels = read_pheme_csv_aug(os.path.join(dir_path, domain))
            train_dataset = CustomDatasetAUG(texts, synonym_texts, labels, tokenizer, max_length)
            train_datasets.append(train_dataset)

    texts, synonym_texts, labels = read_pheme_csv_aug(
        os.path.join(dir_path, '{}.csv'.format(domain_mapping[target_domain])))
    synonym_texts = synonym_replacement(texts)
    test_dataset = CustomDatasetAUG(texts, synonym_texts, labels, tokenizer, max_length)

    return train_datasets, test_dataset


def read_imdb_csv_aug(dir_path):
    data = pd.read_csv(dir_path, header=0)
    texts = data['sentence'].tolist()
    labels = data['labels'].tolist()

    synonym_texts = synonym_replacement(texts)

    return texts, synonym_texts, labels


def load_imdb_dataset_aug(dir_path, tokenizer, max_length):
    val_texts, val_synonym_texts, val_labels = read_imdb_csv_aug(os.path.join(dir_path, 'dev.csv'))
    test_texts, test_synonym_texts, test_labels = read_imdb_csv_aug(os.path.join(dir_path, 'test.csv'))

    val_dataset = CustomDatasetAUG(val_texts, val_synonym_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDatasetAUG(test_texts, test_synonym_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


def read_sst_tsv_aug(dir_path):
    data = pd.read_csv(dir_path, sep='\t', header=0)
    original_texts = data['sentence'].tolist()
    labels = data['label'].tolist()

    texts = []
    for text in original_texts:
        texts.append(text)

    synonym_texts = synonym_replacement(texts)

    return texts, synonym_texts, labels


def load_sst_dataset_aug(dir_path, tokenizer, max_length):
    val_texts, val_synonym_texts, val_labels = read_sst_tsv_aug(os.path.join(dir_path, 'dev.tsv'))
    test_texts, test_synonym_texts, test_labels = read_sst_tsv_aug(os.path.join(dir_path, 'test.tsv'))

    val_dataset = CustomDatasetAUG(val_texts, val_synonym_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDatasetAUG(test_texts, test_synonym_texts, test_labels, tokenizer, max_length)
    return val_dataset, test_dataset


def load_amazon_dataset_aug(dir_path, target_domain, tokenizer, max_length):
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
        synonym_texts = synonym_replacement(texts)
        labels = train_labels[:num_samples]
        train_dataset = CustomDatasetAUG(texts, synonym_texts, labels, tokenizer, max_length)
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

        test_synonym_texts = synonym_replacement(test_texts)
        test_dataset = CustomDatasetAUG(test_texts, test_synonym_texts, test_labels, tokenizer, max_length)

        return train_datasets, test_dataset
    else:
        return train_datasets


def read_nli_csv_aug(dir_path, tokenizer, single_sentence_max_length):
    df = pd.read_csv(dir_path, header=0)
    sentence1 = df['sentence1'].tolist()
    sentence2 = df['sentence2'].tolist()
    labels = df['label'].tolist()

    synonym_sentence1 = synonym_replacement(sentence1)
    synonym_sentence2 = synonym_replacement(sentence2)

    texts = []
    synonym_texts = []
    for i in range(len(sentence1)):
        sentence1_truncation = tokenizer.decode(tokenizer.encode(sentence1[i],
                                                                 truncation=True,
                                                                 max_length=single_sentence_max_length)[1: -1])
        sentence2_truncation = tokenizer.decode(tokenizer.encode(sentence2[i],
                                                                 truncation=True,
                                                                 max_length=single_sentence_max_length)[1: -1])

        text = sentence1_truncation + tokenizer.sep_token + sentence2_truncation
        texts.append(text)

        synonym_sentence1_truncation = tokenizer.decode(tokenizer.encode(synonym_sentence1[i],
                                                                         truncation=True,
                                                                         max_length=single_sentence_max_length)[1: -1])
        synonym_sentence2_truncation = tokenizer.decode(tokenizer.encode(synonym_sentence2[i],
                                                                         truncation=True,
                                                                         max_length=single_sentence_max_length)[1: -1])

        synonym_text = synonym_sentence1_truncation + tokenizer.sep_token + synonym_sentence2_truncation
        synonym_texts.append(synonym_text)

    return texts, synonym_texts, labels


def load_mnli_datasets_aug(dir_path, target_domain, tokenizer, single_sentence_max_length, max_length):
    train_datasets = []
    domains = os.listdir(dir_path)

    for domain in domains:
        if domain != target_domain:
            texts, synonym_texts, labels = read_nli_csv_aug(os.path.join(dir_path, domain, 'train.csv'), tokenizer,
                                                            single_sentence_max_length)
            train_dataset = CustomDatasetAUG(texts, synonym_texts, labels, tokenizer, max_length)
            train_datasets.append(train_dataset)

    if target_domain in domains:
        test_texts, test_synonym_texts, test_labels = read_nli_csv_aug(
            os.path.join(dir_path, target_domain, 'test.csv'),
            tokenizer, single_sentence_max_length)

        test_dataset = CustomDatasetAUG(test_texts, test_synonym_texts, test_labels, tokenizer, max_length)

        return train_datasets, test_dataset
    else:
        return train_datasets


def load_snli_sick_datasets_aug(dir_path, tokenizer, single_sentence_max_length, max_length):
    val_texts, val_synonym_texts, val_labels = read_nli_csv_aug(os.path.join(dir_path, 'dev.csv'),
                                                                tokenizer, single_sentence_max_length)
    test_texts, test_synonym_texts, test_labels = read_nli_csv_aug(os.path.join(dir_path, 'test.csv'),
                                                                   tokenizer, single_sentence_max_length)

    val_dataset = CustomDatasetAUG(val_texts, val_synonym_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDatasetAUG(test_texts, test_synonym_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset


def read_pheme_csv(dir_path):
    data = pd.read_csv(dir_path, header=0)
    texts = data['texts'].tolist()
    labels = data['labels'].tolist()

    return texts, labels


def load_pheme_dataset(dir_path, target_domain, tokenizer, max_length):
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
            texts, labels = read_pheme_csv(os.path.join(dir_path, domain))
            train_dataset = CustomDataset(texts, labels, tokenizer, max_length)
            train_datasets.append(train_dataset)

    texts, labels = read_pheme_csv(
        os.path.join(dir_path, '{}.csv'.format(domain_mapping[target_domain])))
    test_dataset = CustomDataset(texts, labels, tokenizer, max_length)

    return train_datasets, test_dataset


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
    original_texts = data['sentence'].tolist()
    labels = data['label'].tolist()

    texts = []
    for text in original_texts:
        texts.append(text)

    return texts, labels


def load_sst_dataset(dir_path, tokenizer, max_length):
    val_texts, val_labels = read_sst_tsv(os.path.join(dir_path, 'dev.tsv'))
    test_texts, test_labels = read_sst_tsv(os.path.join(dir_path, 'test.tsv'))

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)
    return val_dataset, test_dataset


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
    train_datasets = []
    domains = os.listdir(dir_path)

    for domain in domains:
        if domain != target_domain:
            texts, labels = read_nli_csv(os.path.join(dir_path, domain, 'train.csv'), tokenizer,
                                         single_sentence_max_length)
            train_dataset = CustomDataset(texts, labels, tokenizer, max_length)
            train_datasets.append(train_dataset)

    if target_domain in domains:
        test_texts, test_labels = read_nli_csv(
            os.path.join(dir_path, target_domain, 'test.csv'),
            tokenizer, single_sentence_max_length)

        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

        return train_datasets, test_dataset
    else:
        return train_datasets


def load_snli_sick_datasets(dir_path, tokenizer, single_sentence_max_length, max_length):
    val_texts, val_labels = read_nli_csv(os.path.join(dir_path, 'dev.csv'),
                                         tokenizer, single_sentence_max_length)
    test_texts, test_labels = read_nli_csv(os.path.join(dir_path, 'test.csv'),
                                           tokenizer, single_sentence_max_length)

    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)

    return val_dataset, test_dataset

