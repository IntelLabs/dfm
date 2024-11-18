# PROJECT NOT UNDER ACTIVE MANAGEMENT #  
This project will no longer be maintained by Intel.  
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.  
Intel no longer accepts patches to this project.  
 If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project.  
  
# Anomaly Detection using Feature Reconstruction Error from Deep Features

### Introduction

This repo presents a fast, principled approach for detecting anomalous and out-of-distribution (OOD) samples in deep neural networks (DNN). We propose the application of linear statistical dimensionality reduction techniques on the semantic features produced by a DNN, in order to capture the low-dimensional subspace truly spanned by said features. We show that the feature reconstruction error (FRE), which is the l2-norm of the difference between the original feature in the high-dimensional space and the pre-image of its low-dimensional reduced embedding, is highly effective for OOD and anomaly detection. This dimensionality reduction can be performed using either Principal Component Analysis (PCA), or via a shallow (single layer) linear auto-encoder (AE). In case of the AE, if the weights are shared between the encoder and decoder, i.e $W_{dec} = W_{enc}^T$, then it converges to the same solution as the PCA. Experiments using standard image datasets and DNN architectures demonstrate that our method meets or exceeds best-in-class quality performance, but at a fraction of the computational and memory cost required by the state of the art. It can be trained and run very efficiently.

### Dependencies

Following modules are needed and can be installed using `python -m pip install -r requirements.txt`: 

- numpy  
- scikit-learn  
- PIL  
- torch  
- torchvision  
- pandas

### Dataset

Download the entire MVTec Anomaly Detection dataset (about 4.9 GB) from the [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad) website and store it in a local folder. Recommended default folder location: `./mvtec`. Direct [link](https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz) to dataset.

### Running the code 

To run the code:

```bash
python run_mvtec.py
```

By default, `python run_mvtec.py` will run all 15 object categories in the MVTEC dataset with an *EfficientNet B5* backbone assuming that the dataset is stored in `./mvtec` folder (default location assumed by the script).
    
If the dataset is stored in a different folder, its location can be specified by:

```bash
python run_mvtec.py --dataset_directory <MVTec dataset path>
```
    
Two other backbones are supported: Resnet18 and Resnet50. To run with a different backbone:
    
```bash
python run_mvtec.py --model resnet18
```

Three dimensionality reduction methods are supported: pca, tae (Tied AE), ae (plain AE without shared weights) and any one or more of them can be optionally specified (default: all three modes are run). For example, the command line below will generate results for both PCA and Tied AE. Additionally, the number of training epochs can be optionally be specified if one of the modes being run is a type of AE (default: 250):
    
```bash
python run_mvtec.py --modes pca tae --epochs 200
```

To save the heatmaps used for anomaly segmentation, use the `--save_heatmaps` option and specify an optional output path (default: `.output`):
```bash
python run_mvtec.py --save_heatmaps --output_path results
```
    
To run on GPU:

```bash
python run_mvtec.py --gpu
```
    
To run a specific category from MVTec, provide category name. For example, :

```bash
python run_mvtec.py --object_category hazelnut
```
    
Sample commands using all options

```bash
python run_mvtec.py --dataset_directory <MVTec dataset path> --model resnet18 --object_category hazelnut --gpu --modes pca --save_heatmaps
```

### Reference 
This code is based on the following work. Please cite if you use it.

```tex
@inproceedings{fre2023,
  title={FRE: A Fast Method For Anomaly Detection And Segmentation},
  author={Ibrahima J. Ndiour and
          Nilesh A. Ahuja and
          Utku Genc and
          Omesh Tickoo}
  booktitle={British Machine Vision Conference (BMVC)},
  year={2023},
}
```

<!--
[arxiv version](https://arxiv.org/abs/2203.10422):
```
@misc{https://doi.org/10.48550/arxiv.2203.10422,
  doi = {10.48550/ARXIV.2203.10422},  
  url = {https://arxiv.org/abs/2203.10422},  
  author = {Ndiour, Ibrahima J. and Ahuja, Nilesh A. and Tickoo, Omesh},  
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},  
  title = {Subspace Modeling for Fast Out-Of-Distribution and Anomaly Detection},  
  publisher = {arXiv},  
  year = {2022},  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
-->

<!-- Reviewed 9/11/23 michaelbeale-il -->
