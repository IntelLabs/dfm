# -*- coding: utf-8 -*-

from feature_extractor import FeatureExtractor
import torch
import numpy as np
import argparse
from sklearn.decomposition import PCA
from torchvision.models import resnet18, resnet50, efficientnet_b5
from mvtec import Mvtec
from sklearn import metrics
import pandas as pd

def score(dataloader):
    len_dataset = len(dataloader.dataset)

    scores = np.empty(len_dataset)

    with torch.no_grad():
        count = 0
        for k, data in enumerate(dataloader):
            inputs = data['data'].to(device)
            num_im = inputs.shape[0]
            outputs_inner = feature_extractor(inputs)

            oi = outputs_inner.cpu().numpy()
            scores_pca = np.zeros(oi.shape[0])
            oi_or = oi
            oi_j = pca_model.transform(oi)
            oi_reconstructed = pca_model.inverse_transform(oi_j)
            scores_pca = np.sum(np.square(oi_or - oi_reconstructed), axis=1)
            scores[count: count + num_im] = -scores_pca
            count += num_im

    return scores


def fit(dataloader, pca_threshold):
    eval_loader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=1, shuffle=False)
    data = next(iter(eval_loader))
    features = feature_extractor(data['data'].to(device))
    data_mats_orig = torch.zeros((features.shape[1], len(trainset))).to(device)

    with torch.no_grad():
        data_idx = 0
        for data in dataloader:
            images = data['data'].to(device)
            num_samples = len(images)
            features = feature_extractor(images)
            oi = torch.squeeze(features)
            data_mats_orig[:, data_idx:data_idx+num_samples] = oi.transpose(1, 0)
            data_idx += num_samples

    data_mats_orig = data_mats_orig.cpu().numpy()
    pca_model = PCA(pca_threshold)
    pca_model.fit(data_mats_orig.T)
    return pca_model




def get_args():
    parser = argparse.ArgumentParser(description="Fit a distribution to the deep features of a trained network using"
                                                 "training samples.")
    parser.add_argument("-m", "--model", help="Model to be tested. Default: efficientnet_b5", choices=['resnet18', 'resnet50', 'efficientnet_b5'], default='efficientnet_b5')
    parser.add_argument("--object_category", help="(Optional) MVTec object category. Either name of category, e.g. bottle, cable, etc. or 'all'. Default: all", default='all')
    parser.add_argument("--gpu", help="(Optional) Run on GPU ", action="store_true")
    parser.add_argument("--dataset_directory", help="(Optional) Specify directory of MVTec dataset. Default: ./mvtec", default='./mvtec')
    parser.add_argument("--pca", help="(Optional) The amount of variance that needs to be retained by PCA", type=float, default=0.97)

    args = parser.parse_args()
    return args


args = get_args()

mvtec_categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

model_name = args.model
object_categories = mvtec_categories if args.object_category == 'all' else [args.object_category]
dataset_directory = args.dataset_directory
pca_threshold = args.pca
device = "cuda:0" if args.gpu == True else "cpu"



if args.model == 'resnet18':
    net = resnet18(pretrained=True)
    layer = 'layer3'
    pool_factor = 2
    im_size = 256
elif args.model == 'resnet50':
    net = resnet50(pretrained=True)
    layer = 'layer3'
    pool_factor = 2
    im_size = 256
elif args.model == 'efficientnet_b5':
    net = efficientnet_b5(pretrained=True)
    layer = 'features.6'
    pool_factor = 2
    im_size = 456

net = net.to(device)
net.eval()
feature_extractor = FeatureExtractor(net, layer_name=layer, pool_factor=pool_factor)

auc_roc = list()
auc_pr = list()

for object_category in object_categories:
    print('Processing', object_category)

    trainset = Mvtec(root_dir=dataset_directory, object_type=object_category, split='train', im_size=im_size)
    testset = Mvtec(root_dir=dataset_directory, object_type=object_category, split='test', defect_type='good', im_size=im_size)
    outset = Mvtec(root_dir=dataset_directory, object_type=object_category, split='test', defect_type='defect', im_size=im_size)

    batch_size = 64  # Change if needed

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    outloader = torch.utils.data.DataLoader(outset, batch_size=batch_size, shuffle=False)


    print('>> Training FRE model...')
    pca_model = fit(dataloader=trainloader, pca_threshold=pca_threshold)

    print('>> Evaluating test set')
    scores_test = score(testloader)
    scores_out = score(outloader)

    scores_concat = np.concatenate((scores_test, scores_out))

    ground_truth_out = np.zeros(len(scores_out))
    ground_truth_test = np.ones(len(scores_test))
    ground_truth_concat = np.concatenate((ground_truth_test, ground_truth_out))

    fpr, tpr, _ = metrics.roc_curve(ground_truth_concat, scores_concat)
    precision, recall, _ = metrics.precision_recall_curve(ground_truth_concat, scores_concat)
    auc_roc.append(metrics.auc(fpr, tpr))
    auc_pr.append(metrics.auc(recall, precision))

df = pd.DataFrame({'AUROC': auc_roc, 'AUPR': auc_pr}, index=object_categories)
print()
print(df)
