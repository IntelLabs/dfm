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

class custom_resize():
    """Pre-processing for MNIST inference.
    """

    def __init__(self, im_size, normalize_mean = [0.5, 0.5, 0.5], normalize_std = [0.5, 0.5, 0.5]):
        normalize_tf = TF.Normalize(mean=normalize_mean, std=normalize_std)
        self.tf = TF.Compose([TF.Resize(tuple(im_size), interpolation=TF.InterpolationMode.LANCZOS), TF.ToTensor(), normalize_tf])

    def __call__(self, img):
        return self.tf(img)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


class Mvtec(VisionDataset):

    def __init__(self, root_dir, object_type=None, split=None, defect_type=None, im_size=None):

        super().__init__(root_dir)

        if split == 'train':
            defect_type = 'good' 

        image_dirs = list()
        split_dir = Path(root_dir, object_type, split)
        if split == 'test' and defect_type == 'defect':
            for d in split_dir.iterdir():
                if 'good' in str(d):
                    continue
                image_dirs.append(d)
        else:
            d = split_dir / defect_type
            image_dirs.append(d)

        self.image_list = list()
        for d in image_dirs:
            self.image_list.extend(list(d.glob('*.png')))
        
        self.im_size = (256, 256) if im_size is None else (im_size, im_size)
        self.transform = custom_resize(self.im_size, normalize_mean=imagenet_mean, normalize_std=imagenet_std)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = self.transform(image)
        sample = {'data': image}

        return sample

    def get_transform(self):
        return self.transform

