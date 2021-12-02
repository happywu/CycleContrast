# ------------------------------------------------------------------------------
# Copyright (c) by contributors 
# Licensed under the MIT License.
# Written by Haiping Wu
# ------------------------------------------------------------------------------
import os
import glob
import random

from PIL import Image
import numpy as np
from skimage import color
import scipy.io as sio
import tqdm
import pickle
from collections import OrderedDict, defaultdict

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import multiprocessing
from torch.multiprocessing import Pool


import cycle_contrast.loader


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageNetVal(torch.utils.data.Dataset):
    # the class name and idx do not necessarily follows the standard one
    def __init__(self, root, class_names, class_to_idx, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        # super(ImageNetVal, self).__init__(root, transform=transform,
        #                                     target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        samples = self._make_dataset(class_names, class_to_idx)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 samples"))

        self.loader = default_loader

        self.classes = list(class_names)
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _make_dataset(self, class_names, class_to_idx):
        meta_file = os.path.join(self.root, 'meta_clsloc.mat')
        meta = sio.loadmat(meta_file, squeeze_me=True)['synsets']
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(', ')) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}

        annot_file = os.path.join(self.root, 'ILSVRC2012_validation_ground_truth.txt')
        with open(annot_file, 'r') as f:
            val_idcs = f.readlines()
        val_idcs = [int(val_idx) for val_idx in val_idcs]
        pattern = os.path.join(self.root, 'ILSVRC2012_val_%08d.JPEG')
        samples = []
        for i in range(50000):
            # filter class names needed
            gt_wnid = idx_to_wnid[val_idcs[i]]
            if gt_wnid in class_names:
                samples.append([pattern%(i+1), class_to_idx[gt_wnid]])
        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)



class ImageFolderInstance(torch.utils.data.Dataset):
    """Folder datasets which returns the index of the image as well
    """
    def __init__(self, root, transform=None, target_transform=None):
        self.dataset = datasets.ImageFolder(root, transform)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

SAMPLE_NAME = "AA/AA2pFq9pFTA_000001.jpg"
LEN_SAMPLE_NAME = len(SAMPLE_NAME)
LEN_VID_NAME = len("AA2pFq9pFTA")
LEN_NUM_NAME = len("000001")
LEN_CLIP_NAME = len("0001")

class R2V2Dataset(torch.utils.data.Dataset):
    """Folder datasets which returns the index of the image as well
    """
    def __init__(self, root, transform=None, target_transform=None, data_split='train', return_all_video_frames=False,
                 num_of_sampled_frames=-1, return_same_frame_indicator=False, return_neg_frame=False):
        import pandas as pd
        self.root = root
        self.transform = transform
        self.two_crops_transform = cycle_contrast.loader.TwoCropsTransform(transform)
        self.target_transform = target_transform
        self.data_split = data_split
        self.return_all_video_frames = return_all_video_frames
        self.num_of_sampled_frames = num_of_sampled_frames
        self.return_same_frame_indicator = return_same_frame_indicator
        self.return_neg_frame = return_neg_frame

        self._get_annotations()

        self.loader = default_loader

    @staticmethod
    def get_video_name(name):
        return name[-LEN_NUM_NAME - LEN_VID_NAME - 5: -LEN_NUM_NAME - 5]

    @staticmethod
    def get_frame_id(name):
        return int(name[-LEN_NUM_NAME - 4: -4])

    def get_image_paths(self):
        print('path ############', self.data_basepath)
        return sorted(list(tqdm.tqdm(glob.iglob(os.path.join(self.data_basepath, "*/*.jpg")))))

    def get_image_name(self, key: str, ind: int):
        return os.path.join(self.data_split_path, key[:2], key + "_%06d.jpg" % ind)

    def video_id_frame_id_split(self, name):
        return self.get_video_name(name), self.get_frame_id(name)

    def _get_single_frame(self, path_key, ind):
        return self.transform(self.loader(self.get_image_name(path_key, ind)))

    def _get_annotations(self):
        self.data_basepath = self.root
        self.data_split_path = os.path.join(self.data_basepath)
        pickle_path = os.path.join(self.data_basepath, self.data_split+ "_names.pkl")
        if not os.path.exists(pickle_path):
            print('creat new cache')
            images = self.get_image_paths()
            path_info = OrderedDict()
            video_names = sorted([self.video_id_frame_id_split(name) for name in images])
            for vid_id, ind in video_names:
                if vid_id not in path_info:
                    path_info[vid_id] = []
                path_info[vid_id].append(ind)
            path_info = sorted([(key, val) for key, val in path_info.items()])
            os.makedirs(self.data_split_path, exist_ok=True)
            pickle.dump(path_info, open(pickle_path, "wb"))
        self.path_info = pickle.load(open(pickle_path, "rb"))
        num_frames = int(np.sum([len(p_info[1]) for p_info in self.path_info]))
        print("Num for %s videos %d frames %d" % (self.data_split, len(self.path_info), num_frames))

    def __getitem__(self, index):

        path_key, frame_ids = self.path_info[index]
        target = index

        if self.return_neg_frame:
            ind = np.random.choice(frame_ids, 2, False)
            ind, neg_ind = ind
        else:
            ind = np.random.choice(frame_ids, 1)
        path = self.get_image_name(path_key, ind)
        image = self.loader(path)
        if self.transform is not None:
            sample = self.two_crops_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.num_of_sampled_frames != -1:
            sampled_frame_ids = np.random.choice(frame_ids, self.num_of_sampled_frames, False)
            video_frames = [self.loader(self.get_image_name(path_key, _ind))
                            for _ind in sampled_frame_ids]
            video_frames = [self.transform(video_frame) for video_frame in video_frames]
            if self.return_neg_frame:
                if neg_ind not in sampled_frame_ids:
                    video_frames_neg = self.loader(self.get_image_name(path_key, neg_ind))
                    video_frames_neg = self.transform(video_frames_neg)
                    video_frames.append(video_frames_neg)
                else:
                    idx = np.where(sampled_frame_ids == neg_ind)[0][0]
                    video_frames.append(video_frames[idx])

            video_frames = torch.stack(video_frames, dim=0)

            return sample, target, index, video_frames
        elif self.return_all_video_frames:
            video_frames = [self.loader(self.get_image_name(path_key, _ind))
                            for _ind in frame_ids if _ind != ind]
            video_frames = [self.transform(video_frame) for video_frame in video_frames]
            video_frames = torch.stack([self.transform(image), *video_frames], dim=0)
            return sample, target, index, video_frames
        else:
            return sample, target, index

    def __len__(self):
        # path_info: dictionary; video_name, frame_ids in video
        return len(self.path_info)


def parse_file(dataset_adr, categories):
    dataset = []
    with open(dataset_adr) as f:
        for line in f:
            line = line[:-1].split("/")
            category = "/".join(line[2:-1])
            file_name = "/".join(line[2:])
            if not category in categories:
                continue
            dataset.append([file_name, category])
    return dataset


def get_class_names(path):
    classes = []
    with open(path) as f:
        for line in f:
            categ = "/".join(line[:-1].split("/")[2:])
            classes.append(categ)
    class_dic = {classes[i]: i for i in range(len(classes))}
    return class_dic

