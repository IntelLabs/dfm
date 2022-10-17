# Anomaly Detection using Feature Reconstruction Error from Deep Features

### Introduction

This repo presents a fast, principled approach for detecting anomalous and out-of-distribution (OOD) samples in deep neural networks (DNN). We propose the application of linear statistical dimensionality reduction techniques on the semantic features produced by a DNN, in order to capture the low-dimensional subspace truly spanned by said features. We show that the feature reconstruction error (FRE), which is the l2-norm of the difference between the original feature in the high-dimensional space and the pre-image of its low-dimensional reduced embedding, is highly effective for OOD and anomaly detection. To generalize to intermediate features produced at any given layer, we extend the methodology by applying nonlinear kernel-based methods. Experiments using standard image datasets and DNN architectures demonstrate that our method meets or exceeds best-in-class quality performance, but at a fraction of the computational and memory cost required by the state of the art. It can be trained and run very efficiently.

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
python run_mvtec.py --dataset_directory <MVTec dataset path> --model resnet18 --object_category hazelnut --gpu
```

### Reference 
This code is based on the following work. Please cite if you use it.

```tex
@inproceedings{fre2022,
  title={Subspace Modeling for Fast Out-Of-Distribution and Anomaly Detection},
  author={Ibrahima J. Ndiour and
          Nilesh A. Ahuja and
          Omesh Tickoo}
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  year={2022},
  organization={IEEE}
}
```
