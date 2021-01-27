
import torch
import utils.audio_utils as au
import datasets.transforms as T
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
import wave
import contextlib

class BatAnnotationDataSet(Dataset):
    def __init__(self, audio_file, ann_file, transform=None, return_masks = None):
        self.bat_anns = json.load(open(ann_file))
        self.root_dir = audio_file
        self.transform = transform
        self.prepare = BatConvert(return_masks)

    def __len__(self):
        return len(self.bat_anns)

    def __getitem__(self, idx):

        wav_name = self.bat_anns[idx]['id']
        anns = self.bat_anns[idx]
        spec, sampling_rate, spec_duration = get_spectrogram_sampling_rate(self.root_dir + wav_name)
        
        fname = self.root_dir + wav_name
        with contextlib.closing(wave.open(fname,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
      
        anns_simplified = []

        for ann in anns['annotation']:
            d = {}

            ### GROUND TRUTH BBOX ANNOTATIONS
            width = (ann['end_time'] - ann['start_time']) * (spec.shape[1]/duration)
            height = (ann['high_freq'] - ann['low_freq']) * (spec.shape[0]/150000)
            x = (ann['start_time']) * (spec.shape[1]/duration)
            y = (ann['low_freq']) * (spec.shape[0]/150000)
            d['bbox'] = [x, y, width, height]

            ### AREA = W x H
            area = width * height
            d['area'] = area

            ### CATEGORY ID IS 1 FOR NOT, WILL BE CHANGED LATER
            category_id = 1
            d['category_id'] = category_id

            anns_simplified.append(d)

        target = {'image_id': idx, 'annotations': anns_simplified, 'sampling_rate': sampling_rate}

        spec, target = self.prepare(spec, target)

        if self.transform:
            spec, target = self.transform(spec, target)
            
        return spec, target, sampling_rate, wav_name


class BatConvert(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):

        h, w = image.shape[-2:]
        
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        #boxes[:, 0::2].clamp_(min=0, max=w)
        #boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])


        return image, target

class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, spec, target):
        init_w = spec.shape[2]
        init_h = spec.shape[1]

        x_shr = self.output_size[1]/init_w
        y_shr = self.output_size[0]/init_h

        spec_n = torch.nn.functional.interpolate(spec.unsqueeze(0), size = self.output_size, mode='bilinear', align_corners=False)[0]
        for i, bbox in enumerate(target["boxes"]):
            bbox[0] = bbox[0] * x_shr
            bbox[1] = bbox[1] * y_shr
            bbox[2] = bbox[2] * x_shr
            bbox[3] = bbox[3] * y_shr
            target["area"][i] = bbox[2] * bbox[3]
        
        return spec_n, target       

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

def make_bat_transforms(image_set):
    if image_set == 'train_val':
        return T.Compose([T.ToTensor(), T.Normalize([0.058526332422855], [0.1667903737826997]), Resize((256, 512))])
    #if image_set == 'test':
        #return Compose([ToTensor(), Normalize([0.058526332422855], [0.1667903737826997])])

def build(image_set, args):

    CWD = os.getcwd()
    PATHS = {
        
        ### LOCAL COMPUTER PATH
        "train_val": ('C:/Users/ehopl/Desktop/bat_data/annotations/train_val.json', 'C:/Users/ehopl/Desktop/bat_data/audio/mc_2018/audio/'),
        "test": ('C:/Users/ehopl/Desktop/bat_data/annotations/test.json', 'C:/Users/ehopl/Desktop/bat_data/audio/mc_2019/audio/'),

        ### GPU CLUSTER PATH
        #"train_val": ('/home/s1764306/data/annotations/train_val.json', '/home/s1764306/data/audio/mc_2018/audio/'),
        #"test": ('/home/s1764306/data/annotations/test.json', '/home/s1764306/data/audio/mc_2019/audio/'),
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
