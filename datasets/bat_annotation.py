
from pathlib import Path

import torch
import torch.utils.data
import torchvision

import datasets.transforms as T
import torchvision.transforms.functional as F
from generate_spectrogram import get_spectrogram_sampling_rate

import numpy as np
import json
import os

class BatDetection(torchvision.datasets.CocoDetection):
    def __init__(self, audio_folder, ann_file, transforms, return_masks):
        super(BatDetection, self).__init__(audio_folder, ann_file)
        self._transforms = transforms
        self.prepare = BatConvert(return_masks)
        
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
            
        img, sampling_rate, spec_duration = get_spectrogram_sampling_rate(os.path.join(self.root, path))
             
        image_id = self.ids[index]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


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
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

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

# class FixedResize(object):
#     def __call__(self,spec):
#         if spec.shape[1] == 256:
#             if spec.shape[2] < 1718:
#                 target_tensor = torch.zeros(1, spec.shape[1], 1718)
#                 target_tensor[:, :, :spec.shape[2]] = spec
#                 spec = target_tensor
#             elif spec.shape[2] > 1718:
#                 spec = spec[:,:,:1718]
#             else:
#                 pass
#         return spec

def make_bat_transforms(image_set):
    if image_set == 'train':
        return T.Compose([T.ToTensor(), Resize((256, 512)), T.Normalize([0.058526332422855], [0.1667903737826997])])
    if image_set == 'test':
        return T.Compose([T.ToTensor(), Resize((256, 512)), T.Normalize([0.050141380321473965], [0.3160132308495623])])

def build(image_set, args):

    CWD = os.getcwd()
    PATHS = {
        
        ### LOCAL COMPUTER PATH
        #"train_val": ('C:/Users/ehopl/Desktop/bat_data/annotations/train_val.json', 'C:/Users/ehopl/Desktop/bat_data/audio/mc_2018/audio/'),
        #"test": ('C:/Users/ehopl/Desktop/bat_data/annotations/test.json', 'C:/Users/ehopl/Desktop/bat_data/audio/mc_2019/audio/'),

        ### GPU CLUSTER PATH
        "train": ('/home/s1764306/data/annotations/coco_v_train.json', '/home/s1764306/data/audio/mc_2018/audio/'),
        "test": ('/home/s1764306/data/annotations/coco_v_test.json', '/home/s1764306/data/audio/mc_2019/audio/'),

        "train_b": ('/home/s1764306/data/annotations/coco_v_train_b.json', '/home/s1764306/data/audio/mc_2018/audio/'),
        "test_b": ('/home/s1764306/data/annotations/coco_v_test_b.json', '/home/s1764306/data/audio/mc_2019/audio/'),
    }

    if image_set == 'train':
        if args.bigger_bbox == "True":
            ann_file, audio_file = PATHS['train_b']
            dataset = BatDetection(ann_file = ann_file, audio_folder= audio_file, transforms=make_bat_transforms(image_set), return_masks = False)
        else:
            ann_file, audio_file = PATHS['train']
            dataset = BatDetection(ann_file = ann_file, audio_folder= audio_file, transforms=make_bat_transforms(image_set), return_masks = False)
        return dataset

    elif image_set == 'test':
        if args.bigger_bbox == "True":
            ann_file, audio_file = PATHS['test_b']
            dataset = BatDetection(ann_file = ann_file, audio_folder= audio_file, transforms=make_bat_transforms(image_set), return_masks = False)
        else:
            ann_file, audio_file = PATHS['test']
            dataset = BatDetection(ann_file = ann_file, audio_folder= audio_file, transforms=make_bat_transforms(image_set), return_masks = False)
        return dataset
        
    else:
        return None
