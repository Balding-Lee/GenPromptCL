'''
@Time : 2024/3/19 16:15
@Auth : Qizhi Li
'''
import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import heapq
import pyprind
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return input_ids, attention_mask


class CustomDatasetSentiment(Dataset):
    def __init__(self, data, labels, tokenizer, max_length):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        labels = int(self.labels[index])

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return input_ids, attention_mask, labels


def load_amazon_dataset(dir_path, domain, tokenizer, max_length):
    texts = []

    with open(os.path.join(dir_path, domain, 'all_data.txt'), 'r') as f:
        lines = f.readlines()

    for text in lines:
        line = text.strip().split(' ||| ')
        if len(line) == 2:
            texts.append(line[0])

    dataset = CustomDataset(texts, tokenizer, max_length)

    return dataset


def load_amazon_dataset_sentiment(dir_path, domain, tokenizer, max_length):
    texts, labels = [], []

    with open(os.path.join(dir_path, domain, 'all_data.txt'), 'r') as f:
        lines = f.readlines()

    for text in lines:
        line = text.strip().split(' ||| ')
        if len(line) == 2:
            texts.append(line[0])
            labels.append(line[1])

    dataset = CustomDatasetSentiment(texts, labels, tokenizer, max_length)

    return dataset


def load_sst_dataset_sentiment(dir_path, tokenizer, max_length):
    data = pd.read_csv(dir_path, sep='\t', header=0)
    texts = data['sentence'].tolist()
    labels = data['label'].tolist()

    dataset = CustomDatasetSentiment(texts, labels, tokenizer, max_length)

    return dataset


def show_tnse():
    arrays = []
    domains = []
    for pkl_name in os.listdir('../results/words_hidden_states'):
        domains.append(pkl_name.split('_')[0])
        with open(os.path.join('../results/words_hidden_states', pkl_name), 'rb') as f:
            arrays.append(pickle.load(f))

    labels = np.concatenate([[i] * len(arr) for i, arr in enumerate(arrays)])

    data = []
    for arr in arrays:
        for h in arr.values():
            data.append(h)

    data = np.array(data)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(data)

    colors = ['#F9C08A', '#EDA1A4', '#B3D8D5', '#A4CB9E']

    for i, label in enumerate(np.unique(labels)):
        mask = (labels == label)
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], color=colors[i], label=f'{domains[i]}', s=1)

    plt.legend()
    plt.savefig('../results/word_distribution.pdf')
    plt.show()

    # =========== Check the distribution of out-of-domain words
    out_domain_words = []

    for word in arrays[0].keys():
        if word in arrays[1].keys() and word in arrays[2].keys() and word in arrays[3].keys():
            out_domain_words.append(word)

    data = []
    labels = []
    i = 0
    for arr in arrays:
        for word in arr.keys():
            if word in out_domain_words:
                data.append(arr[word])
                labels.append(i)
        i += 1

    data = np.array(data)
    labels = np.array(labels)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(data)

    colors = ['#F9C08A', '#EDA1A4', '#B3D8D5', '#A4CB9E']

    for i, label in enumerate(np.unique(labels)):
        mask = (labels == label)
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], color=colors[i], label=f'{domains[i]}', s=1)

    plt.legend()
    plt.savefig('../results/out_domain_word_distribution.pdf')
    plt.show()


    # =========== Check the distribution of CLS in different domains
    arrays = []
    domains = []
    for file in os.listdir('../results/cls_hidden_states/'):
        if file.split('_')[1] == 'pos':
            domains.append(file.split('_')[0])
            arrays.append(np.load(os.path.join('../results/cls_hidden_states/', file)))

    labels = np.concatenate([[i] * len(arr) for i, arr in enumerate(arrays)])

    arrays = np.concatenate(arrays, axis=0)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(arrays)

    colors = ['#F9C08A', '#EDA1A4', '#B3D8D5', '#A4CB9E']

    for i, label in enumerate(np.unique(labels)):
        mask = (labels == label)
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], color=colors[i], label=f'{domains[i]}', s=1)

    plt.xticks([])
    plt.yticks([])

    plt.legend(fontsize=12)
    plt.savefig('../results/sst_pos_cls_distribution.pdf')
    plt.show()

    # =========== Examine the distribution of different sentiment CLS in various domains
    category = 'pos'
    arrays = []
    domains = ['sst', 'kitchen', 'electronics', 'dvd', 'book']
    file_paths = ['../results/cls_hidden_states/sst_{}_cls_196.npy'.format(category),
                  '../results/cls_hidden_states/kitchen_{}_cls_196.npy'.format(category),
                  '../results/cls_hidden_states/electronics_{}_cls_196.npy'.format(category),
                  '../results/cls_hidden_states/dvd_{}_cls_196.npy'.format(category),
                  '../results/cls_hidden_states/book_{}_cls_196.npy'.format(category)]

    for file_path in file_paths:
        arrays.append(np.load(file_path))

    labels = np.concatenate([[i] * len(arr) for i, arr in enumerate(arrays)])

    arrays = np.concatenate(arrays, axis=0)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(arrays)

    colors = ['#F48C8C', '#F3F1AD', '#EAFFD0', '#95E1D3', '#F4D8A5']
    edgecolors = ['#f38181', '#FCE38A', '#C0F0D2', '#87CDC0', '#F4BF9D']

    plt.figure(figsize=(10, 10))
    for i, label in enumerate(np.unique(labels)):
        mask = (labels == label)
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], color=colors[i], edgecolor=edgecolors[i], label=f'{domains[i]}')

    plt.xticks([])
    plt.yticks([])

    plt.legend(fontsize=18, framealpha=0.5)
    plt.savefig('../results/sst_{}_cls_distribution_196.pdf'.format(category))
    plt.show()



def count_words():
    model_path = r'D:\huggingface\roberta-base'
    dir_path = '../datasets/amazon'
    domains = ['book', 'dvd', 'electronics', 'kitchen']
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    for domain in tqdm.tqdm(domains):
        num_words = {}
        with open(os.path.join(dir_path, domain, 'all_data.txt'), 'r') as f:
            lines = f.readlines()

        for text in lines:
            line = text.strip().split(' ||| ')
            if len(line) == 2:
                text = line[0]
                words = tokenizer.encode(text)[1: -1]
                # words = text.split(' ')
                for word in words:
                    if num_words.__contains__(word):
                        num_words[word] += 1
                    else:
                        num_words[word] = 1

        with open('../results/words_counter/{}_words_counter.json'.format(domain), 'w') as f:
            json.dump(num_words, f)


def calculate_co_occur_words():
    k = 100
    domains = ['book', 'dvd', 'electronics', 'kitchen']
    word_list = []
    for domain in domains:
        with open('../results/words_counter/{}_words_counter.json'.format(domain), 'r') as f:
            num_words = json.load(f)
            word_list.append(num_words)

    words_count = {}
    for word in word_list[0].keys():
        if all(word in d for d in word_list[1:]):
            words_count[word] = sum(d[word] for d in word_list)

    top_k_words = heapq.nlargest(k, words_count.items(), key=lambda item: item[1])
    ids = []
    for word_id in top_k_words:
        ids.append(int(word_id[0]))

    with open('../results/words_counter/top_{}_ids.pkl'.format(k), 'wb') as f:
        pickle.dump(ids, f)


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model_path = '/home/liqizhi/huggingface/roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    bert_config = RobertaConfig.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path, config=bert_config).to(device)

    dir_path = '../datasets/amazon'
    domains = ['book', 'dvd', 'electronics', 'kitchen']
    max_length = 512

    combined_matrix = []
    combined_names = []
    colors = ['#F9C08A', '#EDA1A4', '#B3D8D5', '#A4CB9E']
    combined_colors = []
    labels = []
    for i, domain in enumerate(domains):
        with open('../results/cls_hidden_states/{}_100_word_hidden_state.json'.format(domain), 'r') as f:
            word_hidden_states = json.load(f)

        domain_states = []
        for name, value in word_hidden_states.items():
            combined_names.append(tokenizer.decode(int(name)).rstrip(' '))
            domain_states.append(value)
            combined_colors.append(colors[i])
            labels.append(domains[i])

        domain_states = np.array(domain_states)
        combined_matrix.append(domain_states)

    combined_matrix = np.vstack(combined_matrix)
    tsne = TSNE(n_components=2, random_state=0)
    reduced_matrix = tsne.fit_transform(combined_matrix)

    reduced_matrix1 = reduced_matrix[:100]
    reduced_matrix2 = reduced_matrix[100:200]
    reduced_matrix3 = reduced_matrix[200:300]
    reduced_matrix4 = reduced_matrix[300:400]

    plt.figure(figsize=(15, 15))
    plt.scatter(reduced_matrix1[:, 0], reduced_matrix1[:, 1], c='#F9C08A', label=domains[0])
    plt.scatter(reduced_matrix2[:, 0], reduced_matrix2[:, 1], c='#EDA1A4', label=domains[1])
    plt.scatter(reduced_matrix3[:, 0], reduced_matrix3[:, 1], c='#B3D8D5', label=domains[2])
    plt.scatter(reduced_matrix4[:, 0], reduced_matrix4[:, 1], c='#A4CB9E', label=domains[3])

    plt.xticks([])
    plt.yticks([])


    plt.legend(fontsize=36)
    plt.savefig('../results/same_word_different_domains.pdf')
    plt.show()

    # Analyze the distribution of words in different domains
    with torch.no_grad():
        for domain in os.listdir(dir_path):
            domain_words_hidden = {}

            dataset = load_amazon_dataset(dir_path, domain, tokenizer, max_length=256)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            pper = pyprind.ProgPercent(len(dataloader))
            for input_ids, att_mask in dataloader:
                input_ids = input_ids.to(device)
                att_mask = att_mask.to(device)

                outputs = model(input_ids, attention_mask=att_mask)
                hidden_states = outputs.last_hidden_state

                for i in range(len(input_ids)):
                    words = tokenizer.decode(input_ids[i]).split(' ')
                    max_length_word_id = len(words)

                    for j in range(max_length_word_id):
                        word = words[j]

                        if domain_words_hidden.__contains__(word):
                            domain_words_hidden[word] += hidden_states[i][j].detach().cpu().numpy()
                        else:
                            domain_words_hidden[word] = hidden_states[i][j].detach().cpu().numpy()

                pper.update()

            for word in domain_words_hidden.keys():
                domain_words_hidden[word] = domain_words_hidden[word] / len(dataset)


            with open('../results/{}_hidden_state.pkl'.format(domain), 'wb') as f:
                pickle.dump(domain_words_hidden, f)

    # Analyze the distribution of sentences in different domains
    with torch.no_grad():
        for domain in os.listdir(dir_path):
            pos_hidden_state, neg_hidden_state = [], []

            dataset = load_amazon_dataset_sentiment(dir_path, domain, tokenizer, max_length=256)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            pper = pyprind.ProgPercent(len(dataloader))
            for input_ids, att_mask, labels in dataloader:
                input_ids = input_ids.to(device)
                att_mask = att_mask.to(device)

                outputs = model(input_ids, attention_mask=att_mask)
                cls_hidden_state = outputs.pooler_output

                pos_indices = torch.where(labels == 1)[0]
                neg_indices = torch.where(labels == 0)[0]
                pos_hidden_state.append(cls_hidden_state[pos_indices])
                neg_hidden_state.append(cls_hidden_state[neg_indices])

                pper.update()

            pos_hidden_state = torch.cat(pos_hidden_state, dim=0)
            neg_hidden_state = torch.cat(neg_hidden_state, dim=0)
            pos_hidden_state = pos_hidden_state.cpu().detach().numpy()
            neg_hidden_state = neg_hidden_state.cpu().detach().numpy()

            with open('../results/{}_hidden_state.json'.format(domain), 'w') as f:
                json.dump(domain_words_hidden, f)

            np.save('../results/cls_hidden_states/{}_pos_cls.npy'.format(domain), pos_hidden_state)
            np.save('../results/cls_hidden_states/{}_neg_cls.npy'.format(domain), neg_hidden_state)

            with open('../results/amazon/{}_pos_cls.npy'.format(domain), 'w') as f:
                pickle.dump(domain_words_hidden, f)

    # Analyze the distribution differences of sentences with different sentiment in various domains
    with torch.no_grad():
        for domain in os.listdir(dir_path):
            pos_hidden_state = []
            neg_hidden_state = []
            dataset = load_amazon_dataset_sentiment(dir_path, domain, tokenizer, max_length=128)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            # pper = pyprind.ProgPercent(len(dataloader))
            for input_ids, att_mask, labels in tqdm.tqdm(dataloader, desc='domain: {}'.format(domain)):
                input_ids = input_ids.to(device)
                att_mask = att_mask.to(device)

                outputs = model(input_ids, attention_mask=att_mask)
                cls_hidden_state = outputs.pooler_output

                pos_indices = torch.where(labels == 1)[0]
                neg_indices = torch.where(labels == 0)[0]
                pos_hidden_state.append(cls_hidden_state[pos_indices])
                neg_hidden_state.append(cls_hidden_state[neg_indices])

                pper.update()

            pos_hidden_state = torch.cat(pos_hidden_state, dim=0)
            neg_hidden_state = torch.cat(neg_hidden_state, dim=0)
            pos_hidden_state = pos_hidden_state.cpu().detach().numpy()
            neg_hidden_state = neg_hidden_state.cpu().detach().numpy()

            np.save('../results/cls_hidden_states/{}_pos_cls_128.npy'.format(domain), pos_hidden_state)
            np.save('../results/cls_hidden_states/{}_neg_cls_128.npy'.format(domain), neg_hidden_state)


    # The hidden states of sentences with different sentiment orientation in the SST dataset
    with torch.no_grad():
        pos_hidden_state = []
        neg_hidden_state = []

        dataset = load_sst_dataset_sentiment('../datasets/sst/test.tsv', tokenizer, max_length=128)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for input_ids, att_mask, labels in tqdm.tqdm(dataloader, desc='domain: sst'):
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device)

            outputs = model(input_ids, attention_mask=att_mask)
            cls_hidden_state = outputs.pooler_output

            pos_indices = torch.where(labels == 1)[0]
            neg_indices = torch.where(labels == 0)[0]
            pos_hidden_state.append(cls_hidden_state[pos_indices])
            neg_hidden_state.append(cls_hidden_state[neg_indices])

            pper.update()

        pos_hidden_state = torch.cat(pos_hidden_state, dim=0)
        neg_hidden_state = torch.cat(neg_hidden_state, dim=0)
        pos_hidden_state = pos_hidden_state.cpu().detach().numpy()
        neg_hidden_state = neg_hidden_state.cpu().detach().numpy()

        np.save('../results/cls_hidden_states/sst_pos_cls.npy', pos_hidden_state)
        np.save('../results/cls_hidden_states/sst_neg_cls.npy', neg_hidden_state)