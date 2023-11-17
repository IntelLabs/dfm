"""MVTec AD Dataset 

License:
    MVTec AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

Reference:
- Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger:
    The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
    Unsupervised Anomaly Detection; in: International Journal of Computer Vision
    129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.

- Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD —
    A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection;
    in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.
"""

from torchvision.datasets import VisionDataset
from PIL import Image
import torchvision.transforms as TF
import PIL
from pathlib import Path
import torch

class custom_resize():
    """Resize and normalize.
    """

    def __init__(self, im_size, normalize_mean = [0.5, 0.5, 0.5], normalize_std = [0.5, 0.5, 0.5]):
        normalize_tf = TF.Normalize(mean=normalize_mean, std=normalize_std)
        self.tf = TF.Compose([TF.Resize(tuple(im_size), interpolation=TF.InterpolationMode.LANCZOS), TF.ToTensor(), normalize_tf])

    def __call__(self, img):
        return self.tf(img)


class binarize:
    """Binarize a grey-scale image by applying a threshold.
    """

    def __init__(self, threshold, inversion=False):
        self.threshold = threshold
        self.inversion = inversion

    def __call__(self, img):
        if self.inversion == True:
            img = img < self.threshold
        else:
            img = img > self.threshold
        img = img.float()
        return img


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


class Mvtec(VisionDataset):

    def __init__(self, root_dir, object_type=None, split=None, defect_type=None, im_size=None, image_only=True):

        super().__init__(root_dir)

        self.split = split
        self.defect_type = defect_type
        self.image_only = image_only
        self.root_dir = Path(root_dir)


        image_dirs = list()
        gt_maps_dirs = list()
        split_dir = Path(root_dir, object_type, split)
        self.image_list = list()
        self.gt_list = list()

        if split == 'train':
            defect_type = 'good' 
            d = split_dir / defect_type
            image_dirs.append(d)
            for d in image_dirs:
                self.image_list.extend(sorted(list(d.glob('*.png'))))
        elif split == 'test': 
            gt_dir = Path(root_dir, object_type, 'ground_truth')
            if defect_type == 'good':
                d = split_dir / defect_type
                image_dirs.append(d)
                for d in image_dirs:
                    self.image_list.extend(sorted(list(d.glob('*.png'))))
            elif defect_type == 'defect':
                for defect_dir in split_dir.iterdir():
                    defect_subtype = defect_dir.parts[-1]
                    if defect_subtype == 'good':
                        continue
                    d = split_dir / defect_subtype
                    g = gt_dir / defect_subtype
                    image_dirs.append(d)
                    gt_maps_dirs.append(g)
                for d, g in zip(image_dirs, gt_maps_dirs):
                    self.image_list.extend(sorted(list(d.glob('*.png'))))
                    self.gt_list.extend(sorted(list(g.glob('*.png'))))
            else:
                d = split_dir / defect_type
                g = gt_dir / defect_type
                image_dirs.append(d)
                gt_maps_dirs.append(g)
                for d, g in zip(image_dirs, gt_maps_dirs):
                    self.image_list.extend(sorted(list(d.glob('*.png'))))
                    self.gt_list.extend(sorted(list(g.glob('*.png'))))

        
        self.im_size = (256, 256) if im_size is None else (im_size, im_size)
        self.transform = custom_resize(self.im_size, normalize_mean=imagenet_mean, normalize_std=imagenet_std)
        self.gt_transform = TF.Compose([TF.Resize(tuple(self.im_size), interpolation=TF.InterpolationMode.LANCZOS), TF.ToTensor(), binarize(0.5)])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]        
        image = Image.open(img_name)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = self.transform(image)
        
        if len(self.gt_list) > 0:
            gt_img_name = self.gt_list[idx]        
            gt_image = Image.open(gt_img_name)
            gt_image = self.gt_transform(gt_image)
        else:
            # gt_image = torch.Tensor()
            gt_image = torch.zeros(image.shape[-2:])

        sample = {'data': image, 'gt': gt_image}
        return sample
    
    def get_subpath(self, idx):
        img_path = self.image_list[idx]
        return img_path.relative_to(self.root_dir)


    def get_transform(self):
        return self.transform

