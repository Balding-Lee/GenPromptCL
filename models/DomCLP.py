'''
@Time : 2025/11/11 11:16
@Auth : Qizhi Li
'''
import tqdm
import torch
import faiss
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaModel


class BaselineClsModel(nn.Module):
    def __init__(self, model_path, num_outputs):
        super().__init__()
        self.model_config = RobertaConfig.from_pretrained(model_path)
        self.model_config.output_hidden_states = True
        self.model = RobertaModel.from_pretrained(model_path, config=self.model_config)

        self.head = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256)
            )

        self.classifier = nn.Linear(768, num_outputs)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)

        out = outputs.pooler_output

        rep = self.head(out)
        proj = F.normalize(rep, dim=1)
        logits = self.classifier(out)

        return out, rep, proj, logits


def compute_features(eval_loader, model, device):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset), 256).to(device)
    domains = torch.zeros(len(eval_loader.dataset), dtype=torch.int64).to(device)
    labels = torch.zeros(len(eval_loader.dataset), dtype=torch.int64).to(device)
    indices_count = torch.zeros(len(eval_loader.dataset), dtype=torch.int64).to(device)
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
        with torch.no_grad():
            # for i, (index, images, labels_, domains_) in enumerate(tqdm.tqdm(eval_loader)):
            for i, (index, _, _, synonym_ids, synonym_attention_mask, labels_, domains_) in enumerate(tqdm.tqdm(eval_loader)):
                index = index.to(device)
                labels_ = labels_.to(device)
                domains_ = domains_.to(device)
                synonym_ids = synonym_ids.to(device)
                synonym_attention_mask = synonym_attention_mask.to(device)

                out, rep, proj, _ = model(synonym_ids, synonym_attention_mask)
                features[index] = proj

                labels[index] = labels_
                domains[index] = domains_
                indices_count[index] += 1

        bool_idx = torch.where(indices_count != 1)[0]
        if len(bool_idx) > 0:
            features[bool_idx] /= 2
            labels[bool_idx] = (labels[bool_idx] / 2).long()
            domains[bool_idx] = (domains[bool_idx] / 2).long()
            indices_count[bool_idx] -= 1

    return features.cpu(), domains.cpu()


def run_kmeans(x, domains, args, num_train_domains, device, train_domains, cuda):
    print('performing kmeans clustering')
    results = {'text2cluster': [], 'centroids': [], 'density': []}
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
        for seed, num_cluster in enumerate(args.num_cluster):
            k = int(num_cluster)
            temp_im2cluster = torch.zeros(len(x), dtype=torch.long).to(device)
            temp_centroids = torch.zeros(num_train_domains * k, 256).to(device)
            temp_density = torch.zeros(num_train_domains * k).to(device)

            for idx_dom, dom in enumerate(train_domains):
                # intialize faiss clustering parameters
                idx = torch.where(domains == idx_dom)[0]
                x_dom = x[idx]
                d = x_dom.shape[1]
                clus = faiss.Clustering(d, k)
                clus.verbose = True
                clus.niter = 20
                clus.nredo = 5
                clus.seed = seed
                clus.max_points_per_centroid = 1000
                clus.min_points_per_centroid = 10

                res = faiss.StandardGpuResources()
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = False
                cfg.device = int(cuda)
                index = faiss.GpuIndexFlatL2(res, d, cfg)

                clus.train(x_dom, index)

                D, I = index.search(x_dom, 1)  # for each sample, find cluster distance and assignments
                im2cluster = [int(n[0]) for n in I]

                # get cluster centroids
                # (num_cluster, 256)
                centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

                # sample-to-centroid distances for each cluster
                Dcluster = [[] for c in range(k)]
                for im, i in enumerate(im2cluster):
                    Dcluster[i].append(D[im][0])

                # concentration estimation (phi)
                density = np.zeros(k)
                for i, dist in enumerate(Dcluster):
                    if len(dist) > 1:
                        d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                        density[i] = d

                # if cluster only has one point, use the max to estimate its concentration
                dmax = density.max()
                for i, dist in enumerate(Dcluster):
                    if len(dist) <= 1:
                        density[i] = dmax

                density = density.clip(np.percentile(density, 10),
                                       np.percentile(density, 90))  # clamp extreme values for stability
                density = args.temperature * density / density.mean()  # scale the mean to temperature

                # convert to cuda Tensors for broadcast
                centroids = torch.Tensor(centroids).to(device)
                centroids = nn.functional.normalize(centroids, p=2, dim=1)

                im2cluster = [k * idx_dom + x for x in im2cluster]
                im2cluster = torch.LongTensor(im2cluster).to(device)
                density = torch.Tensor(density).to(device)

                temp_im2cluster[idx] = im2cluster
                temp_centroids[torch.arange(k * idx_dom, k * (idx_dom + 1))] = centroids
                temp_density[torch.arange(k * idx_dom, k * (idx_dom + 1))] = density

            results['text2cluster'].append(temp_im2cluster)
            results['centroids'].append(temp_centroids)
            results['density'].append(temp_density)

    return results


class DomCLPLoss(nn.Module):
    def __init__(self, device, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(DomCLPLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, domains, num_domains=3, labels=None, mask=None):

        device = self.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        bsz = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(bsz, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != bsz:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        domains_mask = (domains.repeat(2).unsqueeze(0) == domains.repeat(2).unsqueeze(1))
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(bsz * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits = logits

        # compute log_prob
        exp_logits = torch.exp(logits) * domains_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, bsz).mean()
        return loss
