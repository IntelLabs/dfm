# -*- coding: utf-8 -*-

from feature_extractor import FeatureExtractor
import torch
import torch.nn as nn
import numpy as np
import argparse
from sklearn.decomposition import PCA
from torchvision.models import resnet18, resnet50, efficientnet_b5, wide_resnet50_2
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B5_Weights, Wide_ResNet50_2_Weights
from mvtec import Mvtec
from sklearn import metrics
import pandas as pd
import torchvision.transforms.functional as F
import torch.nn.functional as TF
# from skimage import measure
# from numpy import ndarray
# from statistics import mean
import time
# import intel_extension_for_pytorch as ipex
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import matplotlib
from PIL import Image
from pathlib import Path


class AE(nn.Module):

    def __init__(self, fullSz, projSz) -> None:
        super(AE, self).__init__()
        self.fullSz = fullSz
        self.projSz = projSz
        self.encoder_layer = nn.Linear(fullSz, projSz)
        self.decoder_layer = nn.Linear(projSz, fullSz)

    def encoder(self, input):
        encoded = self.encoder_layer(input)
        return encoded

    def decoder(self, input):
        decoded = self.decoder_layer(input)
        return decoded

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded

class TiedAE(nn.Module):

    def __init__(self, fullSz, projSz, weight=None, bias=None) -> None:
        super().__init__()
        self.fullSz = fullSz
        self.projSz = projSz
        if weight is None:
            self.weight = nn.Parameter(torch.empty(projSz, fullSz))
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            self.weight = nn.Parameter(weight)
        if bias is None:
            self.decoder_bias = nn.Parameter(torch.zeros(fullSz))
            self.encoder_bias = nn.Parameter(torch.zeros(projSz))
        else:
            self.decoder_bias = bias
            self.encoder_bias = torch.matmul(-weight, bias)

    def encoder(self, input):
        encoded = TF.linear(input, self.weight, self.encoder_bias)
        return encoded

    def decoder(self, input):
        decoded = TF.linear(input, self.weight.t(), self.decoder_bias)
        return decoded

    def forward(self, input):
        encoded = TF.linear(input, self.weight, self.encoder_bias)
        decoded = TF.linear(encoded, self.weight.t(), self.decoder_bias)
        return decoded



class fromArray(Dataset):

    def __init__(self, Array):
        super().__init__()
        if isinstance(Array, np.ndarray):
            self.Array = torch.Tensor(Array)
        elif isinstance(Array, torch.Tensor):
            self.Array = Array

    def __len__(self):
        return len(self.Array)

    def __getitem__(self, idx):
        sample = self.Array[idx]
        return sample


def score(dataloader, fre_model):
    len_dataset = len(dataloader.dataset)

    scores = torch.empty(len_dataset)
    heatmaps = torch.Tensor(len_dataset, im_size, im_size)
    ground_truth_maps = torch.Tensor(len_dataset, im_size, im_size)

    with torch.no_grad():
        count = 0
        for k, data in enumerate(dataloader):
            inputs = data['data'].to(device)
            
            num_im = inputs.shape[0]
            features = feature_extractor(inputs)
            feature_shapes = feature_extractor.get_feature_shapes()
            features_reconstructed = fre_model(features)
            

            fre = torch.square(features - features_reconstructed).reshape(feature_shapes)
            fre_map = torch.sum(fre, 1)  # NxCxHxW --> NxHxW                
            fre_score = torch.sum(fre_map, (1,2))  # NxHxW --> N

            scores[count: count + num_im] = fre_score
            heatmaps[count: count + num_im] = F.resize(fre_map, size=(im_size, im_size), interpolation=F.InterpolationMode.BILINEAR, antialias=True)
            ground_truth_maps[count: count + num_im] = torch.squeeze(data['gt'])  # GT maps are single-channel (black & white)
            count += num_im

    output = (scores, heatmaps, ground_truth_maps)

    return output


def fit_pca(dataloader, pca_threshold):
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
    weights = torch.Tensor(pca_model.components_).to(device)
    means = torch.Tensor(pca_model.mean_).to(device)
    fre_model = TiedAE(weights.shape[1], weights.shape[0], weight=weights, bias=means)
    fre_model = fre_model.to(device)
    return fre_model


def fit_ae(dataloader, projSz, mode):
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

    epochs = args.epochs
    batch_size = 64
    fullSz = data_mats_orig.shape[0]
    if mode == 'tae':
        fre_model = TiedAE(fullSz, projSz)
    else:
        fre_model = AE(fullSz, projSz)
    feature_set = fromArray(data_mats_orig.T)
    feature_loader = DataLoader(feature_set, batch_size=batch_size, shuffle=True)
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(fre_model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    fre_model = fre_model.to(device)
    for epoch in tqdm(range(epochs)):
        for data in feature_loader:
            feature_in = data.to(device)
            feature_out = fre_model(feature_in)
            loss = loss_fn(feature_in, feature_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return fre_model


def save_heatmaps(mode, heatmaps_test, heatmaps_out, testset, outset):
    heatmaps_concat = torch.cat((heatmaps_test, heatmaps_out), 0)

    min_val = torch.min(heatmaps_concat)
    max_val = torch.max(heatmaps_concat)
    cm = matplotlib.colormaps['viridis']
    heatmaps_test = 1.1*(heatmaps_test - min_val)/max_val
    for i, hm in enumerate(heatmaps_test):
        heatmap = cm(hm)
        heatmap_image = Image.fromarray((heatmap[:, :, :3] * 255).astype(np.uint8))
        heatmap_relative_path = testset.get_subpath(i)
        heatmap_path = Path(args.output_folder) / mode / heatmap_relative_path
        heatmap_path = heatmap_path.resolve()
        heatmap_folder = heatmap_path.parent
        if not heatmap_folder.exists():
            heatmap_folder.mkdir(parents=True)
        heatmap_image.save(heatmap_path)

    heatmaps_out = 1.1*(heatmaps_out - min_val)/max_val
    for i, hm in enumerate(heatmaps_out):
        heatmap = cm(hm)
        heatmap_image = Image.fromarray((heatmap[:, :, :3] * 255).astype(np.uint8))
        heatmap_relative_path = outset.get_subpath(i)
        heatmap_path = Path(args.output_folder) / mode / heatmap_relative_path
        heatmap_path = heatmap_path.resolve()
        heatmap_folder = heatmap_path.parent
        if not heatmap_folder.exists():
            heatmap_folder.mkdir(parents=True)
        heatmap_image.save(heatmap_path)

def calculate_metrics(scores_test, heatmaps_test, gt_maps_test, scores_out, heatmaps_out, gt_maps_out):
    scores_concat = np.concatenate((scores_test, scores_out))

    ground_truth_out = np.ones(len(scores_out))
    ground_truth_test = np.zeros(len(scores_test))
    ground_truth_concat = np.concatenate((ground_truth_test, ground_truth_out))

    fpr, tpr, _ = metrics.roc_curve(ground_truth_concat, scores_concat)
    precision, recall, _ = metrics.precision_recall_curve(ground_truth_concat, scores_concat)

    im_auroc = metrics.auc(fpr, tpr)
    im_aupr = metrics.auc(recall, precision)

    gt_maps_concat = torch.cat((gt_maps_test, gt_maps_out), 0)
    heatmaps_concat = torch.cat((heatmaps_test, heatmaps_out), 0)
    fpr_pix, tpr_pix, _ = metrics.roc_curve(gt_maps_concat.reshape(-1), heatmaps_concat.reshape(-1))
    pixel_auroc = metrics.auc(fpr_pix, tpr_pix)
    return im_auroc, im_aupr, pixel_auroc
   

def get_args():
    parser = argparse.ArgumentParser(description="Fit a distribution to the deep features of a trained network using"
                                                 "training samples.")
    parser.add_argument("-m", "--model", help="Model to be tested. Default: efficientnet_b5", choices=['resnet18', 'resnet50', 'efficientnet_b5', 'wideresnet50'], default='efficientnet_b5')
    parser.add_argument("--object_categories", help="(Optional) MVTec object category. Either name of category, e.g. bottle, cable, etc. or 'all'. Default: all", default=['all'], nargs='+')
    parser.add_argument("--proj_size", help="(Optional) Latent space dimension of AutoEncoder. Provide either one value per object category or a single value for all.", type=int, nargs='+')
    parser.add_argument("--gpu", help="(Optional) Run on GPU ", action="store_true")
    parser.add_argument("--dataset_directory", help="(Optional) Specify directory of MVTec dataset. Default: ./mvtec", default='./mvtec')
    parser.add_argument("--pca", help="(Optional) The amount of variance that needs to be retained by PCA", type=float, default=0.97)
    parser.add_argument("--epochs", help="(Optional) Number of epochs for training AE", type=int, default=250)
    # parser.add_argument("--calc_pro", action="store_true")
    # parser.add_argument("--ipex", action="store_true")
    parser.add_argument("--modes", help="Choose one or more modes from pca, tae (Tied AE), or ae to run", choices={'pca', 'ae', 'tae'}, nargs='+', default=['pca', 'tae'])
    parser.add_argument("--output_folder")
    parser.add_argument("--save_heatmaps", action="store_true")

    args = parser.parse_args()
    return args


args = get_args()

mvtec_categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
model_name = args.model
object_categories = mvtec_categories if args.object_categories == ['all'] else args.object_categories
proj_sizes = list()
if args.proj_size is not None:    
    if len(args.proj_size) == 1:
        proj_sizes = [args.proj_size[0] for x in range(len(object_categories))]
    elif len(args.proj_size) == len(object_categories):
        proj_sizes = object_categories
    else:
        print(f"ERROR: {len(args.proj_size)} values found for --proj_size, but {len(object_categories)} object categories were provided.")
        sys.exit(1)
else:
    if 'pca' in args.modes:
        print('WARNING: Latent AE dimension not provided, will be inherited from corresponding PCA model.')
    else:
        print('ERROR: Latent AE dimension not provided.')
        sys.exit()

dataset_directory = args.dataset_directory
pca_threshold = args.pca
device = "cuda:0" if args.gpu == True else "cpu"

if args.model == 'resnet18':
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    layer = 'layer3'
    pool_factor = 2
    im_size = 256
elif args.model == 'resnet50':
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    layer = 'layer3'
    pool_factor = 2
    im_size = 256
elif args.model == 'wideresnet50':
    net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    layer = 'layer3'
    pool_factor = 2
    im_size = 256
elif args.model == 'efficientnet_b5':
    net = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
    layer = 'features.6'
    pool_factor = 2
    im_size = 456

net = net.to(device)
net.eval()

# if args.ipex == True and args.gpu == False:
#     net = ipex.optimize(net)

feature_extractor = FeatureExtractor(net, layer_name=layer, pool_factor=pool_factor)

auc_roc_im_pca = list()
auc_roc_pix_pca = list()
auc_roc_pro_pca = list()
auc_roc_im_ae = list()
auc_roc_pix_ae = list()
auc_roc_pro_ae = list()
auc_roc_im_tae = list()
auc_roc_pix_tae = list()
auc_roc_pro_tae = list()

for obj_idx, object_category in enumerate(object_categories):
    print('>>Processing', object_category)

    trainset = Mvtec(root_dir=dataset_directory, object_type=object_category, split='train', im_size=im_size)
    testset = Mvtec(root_dir=dataset_directory, object_type=object_category, split='test', defect_type='good', im_size=im_size)
    outset = Mvtec(root_dir=dataset_directory, object_type=object_category, split='test', defect_type='defect', im_size=im_size)

    batch_size = 64  # Change if needed

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    outloader = torch.utils.data.DataLoader(outset, batch_size=batch_size, shuffle=False, num_workers=2)


    if 'pca' in args.modes:
        print('Training PCA model...')
        train_start = time.time()
        pca_model = fit_pca(dataloader=trainloader, pca_threshold=pca_threshold)
        train_end = time.time()
        print(f'Feature size:{pca_model.weight.shape[1]}, Reduced size:{pca_model.weight.shape[0]}')
        print(f'PCA Training time {train_end - train_start}')
        print('Evaluating test set')
        pred_start = time.time()
        scores_test_pca, heatmaps_test_pca, gt_maps_test_pca = score(testloader, pca_model)
        scores_out_pca, heatmaps_out_pca, gt_maps_out_pca = score(outloader, pca_model)
        pred_end = time.time()
        print(f'PCA Prediction time {pred_end - pred_start}')
        im_auroc_pca, im_aupr_pca, pixel_auroc_pca = calculate_metrics(
            scores_test_pca, heatmaps_test_pca, gt_maps_test_pca, scores_out_pca, heatmaps_out_pca, gt_maps_out_pca
        )
        print(f'PCA: Image AUROC: {im_auroc_pca}, Image AUPR: {im_aupr_pca}, Pixel AUROC: {pixel_auroc_pca}')
        auc_roc_im_pca.append(im_auroc_pca)
        auc_roc_pix_pca.append(pixel_auroc_pca)
        if args.save_heatmaps:
            save_heatmaps('pca', heatmaps_test_pca, heatmaps_out_pca, testset, outset)



    if 'tae' in args.modes:
        print('Training Tied AE model...')
        train_start = time.time()
        if len(proj_sizes) == 0:
            projSz = pca_model.weight.shape[0]
        else:
            projSz = proj_sizes[obj_idx]
        tae_model = fit_ae(dataloader=trainloader, projSz=projSz, mode='tae')
        train_end = time.time()
        print(f'AE Training time {train_end - train_start}')
        pred_start = time.time()
        scores_test_tae, heatmaps_test_tae, gt_maps_test_tae = score(testloader, tae_model)
        scores_out_tae, heatmaps_out_tae, gt_maps_out_tae = score(outloader, tae_model)
        pred_end = time.time()
        print(f'AE Prediction time {pred_end - pred_start}')
        im_auroc_tae, im_aupr_tae, pixel_auroc_tae = calculate_metrics(
            scores_test_tae, heatmaps_test_tae, gt_maps_test_tae, scores_out_tae, heatmaps_out_tae, gt_maps_out_tae
        )
        print(f'AE: Image AUROC: {im_auroc_tae}, Image AUPR: {im_aupr_tae}, Pixel AUROC: {pixel_auroc_tae}')
        auc_roc_im_tae.append(im_auroc_tae)
        auc_roc_pix_tae.append(pixel_auroc_tae)
        if args.save_heatmaps:
            save_heatmaps('tae', heatmaps_test_tae, heatmaps_out_tae, testset, outset)

    if 'ae' in args.modes:
        print('Training plain AE model...')
        train_start = time.time()
        if len(proj_sizes) == 0:
            projSz = pca_model.weight.shape[0]
        else:
            projSz = proj_sizes[obj_idx]
        ae_model = fit_ae(dataloader=trainloader, projSz=projSz, mode='ae')
        train_end = time.time()
        print(f'AE Training time {train_end - train_start}')
        pred_start = time.time()
        scores_test_ae, heatmaps_test_ae, gt_maps_test_ae = score(testloader, ae_model)
        scores_out_ae, heatmaps_out_ae, gt_maps_out_ae = score(outloader, ae_model)
        pred_end = time.time()
        print(f'AE Prediction time {pred_end - pred_start}')
        im_auroc_ae, im_aupr_ae, pixel_auroc_ae = calculate_metrics(
            scores_test_ae, heatmaps_test_ae, gt_maps_test_ae, scores_out_ae, heatmaps_out_ae, gt_maps_out_ae
        )
        print(f'AE: Image AUROC: {im_auroc_ae}, Image AUPR: {im_aupr_ae}, Pixel AUROC: {pixel_auroc_ae}')
        auc_roc_im_ae.append(im_auroc_ae)
        auc_roc_pix_ae.append(pixel_auroc_ae)
        if args.save_heatmaps:
            save_heatmaps('ae', heatmaps_test_ae, heatmaps_out_ae, testset, outset)


results = dict()
if 'pca' in args.modes:
    results['Image AUROC PCA'] = auc_roc_im_pca 
    results['Pixel AUROC PCA'] = auc_roc_pix_pca

if 'ae' in args.modes:
    results['Image AUROC AE'] = auc_roc_im_ae 
    results['Pixel AUROC AE'] = auc_roc_pix_ae

if 'tae' in args.modes:
    results['Image AUROC TAE'] = auc_roc_im_tae 
    results['Pixel AUROC TAE'] = auc_roc_pix_tae
results_df = pd.DataFrame(results, index=object_categories)

print(results_df)
