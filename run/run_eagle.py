'''
@Time : 2024/4/2 11:26
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
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import e_utils
from models import eagle


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    num_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad and 'roberta' not in name and 'classifier' not in name:
            num_parameters += param.numel()
    return num_parameters / 1e6 


def evaluate(model, data_loader, tgt_domain, aug=True):
    preds = []
    labels = []
    loss_total = 0.0
    model.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc='test on {}'.format(tgt_domain)):
            if aug:
                input_ids = batch[0].to(device)
                att_masks = batch[1].to(device)
                tgt = batch[4].to(device)
            else:
                input_ids = batch[0].to(device)
                att_masks = batch[1].to(device)
                tgt = batch[2].to(device)

            logits, _, _ = model(input_ids, att_masks)
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
                                 'state', 'telephone', 'travel', 'sick', 'snli'],
                        default='book')
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--cuda', type=str, default=0)
    parser.add_argument('--model', type=str, default='roberta-base')
    args = parser.parse_args()

    target_domain = args.target_domain
    device = torch.device('cuda:%s' % args.cuda if torch.cuda.is_available() else 'cpu')
    seed = args.seed
    set_seed(seed)

    sa_domains = ['book', 'dvd', 'electronics', 'kitchen', 'imdb', 'sst']
    rumour_domains = ['ch', 'f', 'gw', 'os', 's']
    nli_domains = ['fiction', 'government', 'state', 'telephone', 'travel', 'sick', 'snli']

    model_path = '/sdc1/liqizhi/huggingface/%s' % args.model
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
        if target_domain == 'snli' or target_domain == 'sick':
            num_domains = len(os.listdir(mnli_dir_path)) + 1
        else:
            num_domains = len(os.listdir(mnli_dir_path))
        single_sentence_max_length = 48
        max_length = 128

    save_dir = 'parameters/eagle_{}.bin'.format(target_domain)

    batch_size = 8
    weight_decay = 0.01
    lr = 1e-5  # 0.000015
    num_epochs = 30

    lamb1 = 1
    lamb2 = 0.9
    lamb3 = 0.1

    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    if args.aug:
        if target_domain in sa_domains:
            if target_domain == 'sst' or target_domain == 'imdb':
                train_datasets = e_utils.load_amazon_dataset_aug(amazon_dir_path, target_domain, tokenizer,
                                                                 max_length)

                if target_domain == 'imdb':
                    _, test_dataset = e_utils.load_imdb_dataset_aug(imdb_dir_path, tokenizer,
                                                                              max_length)
                else:
                    _, test_dataset = e_utils.load_sst_dataset_aug(sst_dir_path, tokenizer,
                                                                             max_length)

            else:
                train_datasets, test_dataset = e_utils.load_amazon_dataset_aug(amazon_dir_path, target_domain,
                                                                               tokenizer, max_length)
        elif target_domain in rumour_domains:
            train_datasets, test_dataset = e_utils.load_pheme_dataset_aug(pheme_dir_path, target_domain,
                                                                          tokenizer,
                                                                          max_length)
        elif target_domain in nli_domains:
            if target_domain == 'sick' or target_domain == 'snli':
                train_datasets = e_utils.load_mnli_datasets_aug(mnli_dir_path, target_domain, tokenizer,
                                                                single_sentence_max_length, max_length)
                if target_domain == 'sick':
                    test_dir = sick_dir_path
                else:
                    test_dir = snli_dir_path
                _, test_dataset = e_utils.load_snli_sick_datasets_aug(test_dir, tokenizer,
                                                                                single_sentence_max_length,
                                                                                max_length)
            else:
                train_datasets, test_dataset = e_utils.load_mnli_datasets_aug(mnli_dir_path, target_domain,
                                                                              tokenizer,
                                                                              single_sentence_max_length,
                                                                              max_length)
    else:
        if target_domain in sa_domains:
            if target_domain == 'sst' or target_domain == 'imdb':
                train_datasets = e_utils.load_amazon_dataset(amazon_dir_path, target_domain, tokenizer,
                                                             max_length)

                if target_domain == 'imdb':
                    _, test_dataset = e_utils.load_imdb_dataset(imdb_dir_path, tokenizer, max_length)
                else:
                    _, test_dataset = e_utils.load_sst_dataset(sst_dir_path, tokenizer, max_length)

            else:
                train_datasets, test_dataset = e_utils.load_amazon_dataset(amazon_dir_path, target_domain,
                                                                           tokenizer, max_length)
        elif target_domain in rumour_domains:
            train_datasets, test_dataset = e_utils.load_pheme_dataset(pheme_dir_path, target_domain, tokenizer,
                                                                      max_length)
        elif target_domain in nli_domains:
            if target_domain == 'sick' or target_domain == 'snli':
                train_datasets = e_utils.load_mnli_datasets(mnli_dir_path, target_domain, tokenizer,
                                                            single_sentence_max_length, max_length)
                if target_domain == 'sick':
                    test_dir = sick_dir_path
                else:
                    test_dir = snli_dir_path
                _, test_dataset = e_utils.load_snli_sick_datasets(test_dir, tokenizer,
                                                                            single_sentence_max_length,
                                                                            max_length)
            else:
                train_datasets, test_dataset = e_utils.load_mnli_datasets(mnli_dir_path, target_domain,
                                                                          tokenizer,
                                                                          single_sentence_max_length,
                                                                          max_length)

    train_dataloader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
                        for train_dataset in train_datasets]
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    if target_domain in nli_domains:
        model = eagle.Backbone(model_path, num_outputs=3).to(device)
    else:
        model = eagle.Backbone(model_path).to(device)
    domain_adv = eagle.DomainDiscriminators(num_domains, lr).to(device)
    grl = eagle.GradientReverse().to(device)
    simclr = eagle.SimCLR().to(device)

    model_parameters = count_parameters(model)
    model_parameters += count_parameters(domain_adv)
    model_parameters += count_parameters(grl)
    model_parameters += count_parameters(simclr)
    print('Number of parameters: {:.4f}M'.format(model_parameters))

    criteria = nn.CrossEntropyLoss()

    param_list = [
        {'params': model.parameters()},
        {'params': domain_adv.parameters()},
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
        total_loss = 0.0
        num_train_steps = 0
        start_time = time.time()
        pbar = tqdm.tqdm(zip(*train_dataloader), desc='Target domain {}, Seed {}, Training epoch {}'.format(
            target_domain, seed, epoch))

        for step, batches in enumerate(pbar):
            model.train()

            logits_list, zi_list, zj_list, labels = [], [], [], []
            pos_cls_list, neg_cls_list = [], []

            for idx, batch in enumerate(batches):
                original_input_ids = batch[0].to(device)
                original_att_masks = batch[1].to(device)
                if args.aug:
                    synonym_input_ids = batch[2].to(device)
                    synonym_att_masks = batch[3].to(device)
                    tgt = batch[4].to(device)
                else:
                    tgt = batch[2].to(device)

                if args.aug:
                    input_ids = torch.cat((original_input_ids, synonym_input_ids), dim=0)
                    att_masks = torch.cat((original_att_masks, synonym_att_masks), dim=0)
                else:
                    input_ids = original_input_ids
                    att_masks = original_att_masks

                if args.aug:
                    targets = torch.cat((tgt, tgt), dim=0)
                else:
                    targets = tgt

                logits, z, cls_representations = model(input_ids, att_masks)

                original_cls_representations = cls_representations[:tgt.shape[0], :]
                pos_cls_representations = original_cls_representations[torch.where(tgt > 0)]
                neg_cls_representations = original_cls_representations[torch.where(tgt == 0)]
                pos_cls_representations = grl(pos_cls_representations)
                neg_cls_representations = grl(neg_cls_representations)
                pos_cls_list.append(pos_cls_representations)
                neg_cls_list.append(neg_cls_representations)

                logits_list.append(logits)

                if args.aug:
                    zi_list.append(z[:tgt.shape[0], :])
                    zj_list.append(z[tgt.shape[0]:, :])
                else:
                    zi_list.append(z)
                    zj_list.append(z)

                labels.append(targets)

            pred = torch.cat(logits_list, dim=0)
            # shape: (batch * num_domains, output_size)
            zi = torch.cat(zi_list, dim=0)
            zj = torch.cat(zj_list, dim=0)
            labels = torch.cat(labels, dim=0)

            ce_loss = criteria(pred, labels)
            loss_domain_pos, _, _ = domain_adv(pos_cls_list)
            loss_domain_neg, _, _ = domain_adv(neg_cls_list)
            loss_domain = (loss_domain_pos + loss_domain_neg) * 0.5
            simclr_loss = simclr(zi, zj)

            loss = lamb1 * ce_loss + lamb2 * loss_domain + lamb3 * simclr_loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            num_train_steps += 1

        _, macro_F1 = evaluate(model, test_dataloader, target_domain, args.aug)

        if macro_F1 > best_macro_F1:
            best_macro_F1 = macro_F1
            torch.save(model.state_dict(), save_dir)
        print("Epoch {}, F1: {:.4f}, ".format(epoch, macro_F1))

    if target_domain == 'imdb' or target_domain == 'sst' or target_domain == 'sick' or target_domain == 'snli':
        if target_domain in nli_domains:
            model = eagle.Backbone(model_path, num_outputs=3).to(device)
        else:
            model = eagle.Backbone(model_path).to(device)
        model.load_state_dict(torch.load(save_dir))
        _, test_f1 = evaluate(model, test_dataloader, target_domain, args.aug)

        print('Test F1: {:.4f}'.format(test_f1))
        best_f1 = test_f1
    else:
        print('best macro-F1: {:.4f}'.format(best_macro_F1))
        best_f1 = best_macro_F1

    if not os.path.exists('results/scores/%s/EAGLE/%s' % (task_name, args.model)):
        os.makedirs('results/scores/%s/EAGLE/%s' % (task_name, args.model))

    with open('results/scores/%s/EAGLE/%s/%s_aug-%s.txt' % (task_name, args.model, target_domain, args.aug), 'a') as f:
        f.write('f1: %.4f\n' % best_f1)