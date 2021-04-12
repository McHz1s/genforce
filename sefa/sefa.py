"""SeFa."""

import os
import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict

import torch
import datetime

from models import parse_gan_type
from utils.misc import to_tensor, post_process, load_generator, factorize_weight, HtmlPageVisualizer
from skimage.measure import compare_ssim

class sample_info(object):
    def __init__(self):
        self.semantic_list = []
        self.best_ssim = 0
        self.best_id = 0

    def add(self, img_list, ssim_list):
        self.semantic_list.append({'img_list': img_list, 'ssim_list': ssim_list})
        ssim = sum(ssim_list)/len(ssim_list)
        if ssim >= self.best_ssim:
            self.best_ssim, self.best_id = ssim, len(self.semantic_list) - 1
        return ssim

class Sefa_info(object):
    def __init__(self, distance, values, sem_num, sam_num):
        self.distance = distance
        self.values = values
        self.sample_list = [sample_info() for _ in range(sam_num)]
        self.best_id = 0
        self.best_ssim = 0
        self.sem_num = sem_num
        self.sam_num = sam_num

    def add(self, sam_id, img_list, ssim_list):
        """
        result_dict: {img_list:[], ssim_list:[], ssim:float}
        """
        ssim = self.sample_list[sam_id].add(img_list, ssim_list)
        if ssim > self.best_ssim:
            self.best_id, self.best_ssim = sam_id, ssim

class Sefa(object):
    def __init__(self, cfg, generator, gt_imgs):
        # Factorize weights.
        self.generator = generator
        self.gan_type = parse_gan_type(self.generator)
        self.layers, self.boundaries, self.values = factorize_weight(self.generator, cfg.layer_idx)
        # Set random seed.
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        # Prepare codes.
        codes = torch.randn(cfg.num_samples, self.generator.z_space_dim).cuda()
        if self.gan_type == 'pggan':
            codes = self.generator.layer0.pixel_norm(codes)
        elif self.gan_type in ['stylegan', 'stylegan2']:
            codes = self.generator.mapping(codes)['w']
            codes = self.generator.truncation(codes,
                                              trunc_psi=cfg.trunc_psi,
                                              trunc_layers=cfg.trunc_layers)
        self.codes = codes.detach().cpu().numpy()
        # Generate visualization pages.
        self.distances = np.linspace(cfg.start_distance, cfg.end_distance, cfg.step)
        self.num_sam = cfg.num_samples
        self.num_sem = cfg.num_semantics
        self.gt_imgs = gt_imgs

    def inference(self):
        re_sefa_info = Sefa_info(self.distances, self.values, self.num_sem, self.num_sam)
        for sam_id in tqdm(range(self.num_sam), desc='Sample ', leave=False):
            code = self.codes[sam_id:sam_id + 1]
            for sem_id in tqdm(range(self.num_sem), desc='Semantic ', leave=False):
                boundary = self.boundaries[sem_id:sem_id + 1]
                img_list, ssim_list = [], []
                for col_id, d in enumerate(self.distances, start=1):
                    temp_code = code.copy()
                    if self.gan_type == 'pggan':
                        temp_code += boundary * d
                        image = self.generator(to_tensor(temp_code))['image']
                    elif self.gan_type in ['stylegan', 'stylegan2']:
                        temp_code[:, self.layers, :] += boundary * d
                        image = self.generator.synthesis(to_tensor(temp_code))['image']
                    image = post_process(image, transpose=True)[0]
                    gt_image = self.gt_imgs[col_id]
                    ssim = compare_ssim(image, gt_image, multichannel=True)
                    img_list.append(image)
                    ssim_list.append(ssim)
                re_sefa_info.add(sam_id, img_list, ssim_list)
        return re_sefa_info

