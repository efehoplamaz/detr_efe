
import torch
import utils.audio_utils as au
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms, utils
from skimage import io, transform
from generate_spectrogram import get_spectrogram_sampling_rate, display_spectrogram
from matplotlib import pyplot as plt
import numpy as np
import json
import utils.audio_utils as au
import os

class BatAnnotationDataSet(Dataset):
    def __init__(self, audio_file, ann_file, spec_shape = (256, 512), transform=None, return_masks = None):
        self.bat_anns = json.load(open(ann_file))
        self.root_dir = audio_file
        self.transform = transform
        self.spec_shape = spec_shape
        self.prepare = BatConvert(return_masks)
        self.params = {}
        self.params['fft_win_length'] = 1024 / 441000.0  # 1024 / 441000.0
        self.params['resize_factor'] = 0.5     # resize so the spectrogram at the input of the network
        self.params['fft_overlap']    = 0.75 

    def __len__(self):
        return len(self.bat_anns)

    def __getitem__(self, idx):

        wav_name = self.bat_anns[idx]['id']
        anns = self.bat_anns[idx]
        spec, sampling_rate, spec_duration = get_spectrogram_sampling_rate(self.root_dir + wav_name)
        
        if self.transform:
            spec = self.transform(spec)
            
        spec_n = torch.nn.functional.interpolate(spec.unsqueeze(0), size = self.spec_shape, mode='bilinear', align_corners=False)[0]
        spec_duration_resized = au.x_coords_to_time(spec_n.shape[2], sampling_rate, self.params['fft_win_length'], self.params['fft_overlap'])
        x_shr_coeff = spec_duration_resized/spec_duration
        
        anns_simplified = []
        for ann in anns['annotation']:
            d = {}
            x = ann['start_time'] * x_shr_coeff
            y = ann['high_freq']
            width = (ann['end_time'] - ann['start_time']) * x_shr_coeff
            height = ann['high_freq'] - ann['low_freq']
            category_id = 0
            area = width * height
            d['bbox'] = [x, y, width, height]
            d['area'] = area
            d['category_id'] = category_id
            anns_simplified.append(d)
        target = {'image_id': idx, 'annotations': anns_simplified, 'sampling_rate': sampling_rate}

        spec, target = self.prepare(spec_n, target)
        
        return spec, target


class BatConvert(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):

        c, w, h = image.shape
        
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes

        area = torch.tensor([obj["area"] for obj in anno])
        target["area"] = area
        
        target["image_id"] = image_id 

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

# class Rescale(object):

#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size

#     def __call__(self, spec, target):

#         c, h, w = spec.shape

#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size

#         new_h, new_w = int(new_h), int(new_w)

#         spec = transform.resize(spec[0].numpy(), (new_h, new_w))

#         return torch.from_numpy(spec).unsqueeze(0), target


class ToTensor(object):
    def __call__(self, spec):
        return F.to_tensor(spec)

class FixedResize(object):
    def __call__(self,spec):
        if spec.shape[1] == 256:
            if spec.shape[2] < 1718:
                target_tensor = torch.zeros(1, spec.shape[1], 1718)
                target_tensor[:, :, :spec.shape[2]] = spec
                spec = target_tensor
            elif spec.shape[2] > 1718:
                spec = spec[:,:,:1718]
            else:
                pass
        return spec

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, spec):
        for t in self.transforms:
            spec = t(spec)
        return spec

def make_bat_transforms(image_set):
    if image_set == 'train_val':
        return Compose([ToTensor(), FixedResize()])
    if image_set == 'test':
        return Compose([ToTensor(), FixedResize()])

def build(image_set, args):

    CWD = os.getcwd()
    PATHS = {
        "train_val": ('/home/s1764306/data/annotations/train_val.json', '/home/s1764306/data/audio/mc_2018/audio/'),
        "test": ('/home/s1764306/data/annotations/test.json', '/home/s1764306/data/audio/mc_2019/audio/'),
    }

    if image_set == 'train_val':
        ann_file, audio_file = PATHS['train_val']
        dataset = BatAnnotationDataSet(ann_file = ann_file, audio_file= audio_file, transform=make_bat_transforms(image_set))
        train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
        return train_set, val_set

    elif image_set == 'test':
        ann_file, audio_file = PATHS['test']
        dataset = BatAnnotationDataSet(ann_file = ann_file, audio_file= audio_file, transform=make_bat_transforms(image_set))
        return dataset
    else:
        return None
