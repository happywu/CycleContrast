# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random

import torch


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class MultiCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, num_crops=2):
        self.base_transform = base_transform
        self.num_crops = num_crops

    def __call__(self, x):
        multi_crops = [self.base_transform(x) for _i in range(self.num_crops)]
        return multi_crops


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class data_prefetcher():
    def __init__(self, loader, return_all_video_frames=True):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.return_all_video_frames = return_all_video_frames
        self.preload()

    def preload(self):
        try:
            if self.return_all_video_frames:
                self.next_input, self.next_target, self.next_indices, self.next_video_frames = next(self.loader)
            else:
                self.next_input, self.next_target, self.next_indices = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_indices = None
            self.next_video_frames = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input[0] = self.next_input[0].cuda(non_blocking=True)
            self.next_input[1] = self.next_input[1].cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_indices = self.next_indices.cuda(non_blocking=True)
            if self.return_all_video_frames:
                self.next_video_frames = self.next_video_frames.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        indices = self.next_indices
        if self.return_all_video_frames:
            video_frames = self.next_video_frames
        if input is not None:
            input[0].record_stream(torch.cuda.current_stream())
        if input is not None:
            input[1].record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if indices is not None:
            indices.record_stream(torch.cuda.current_stream())
        if self.return_all_video_frames and video_frames is not None:
            video_frames.record_stream(torch.cuda.current_stream())
        self.preload()
        if self.return_all_video_frames:
            return input, target, indices, video_frames
        else:
            return input, target, indices
