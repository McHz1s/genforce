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

class Sefa_info(object):
    def __init__(self, distance, values):
        self.distance = distance
        self.values = values
        self.sample_list = []
        self.best_id = [0, 0]
        self.best_ssim = 0

    def add_sample(self, result_dict):
        """
        result_dict: {img_list:[], ssim_list:[], ssim:float}
        """
        self.sample_list.append(result_dict)
        if result_dict['ssim'] > self.best_ssim:
            self.best_ssim = result_dict['ssim']
            self.best_id = [len(self.sample_list) - 1, ]

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
        """
        re_dict = {'distance':[](len=step), 'values':[](len=sem_num),
        'sample_list':[{'semantic_img_list':[img_num], 'ssim':[img_num]}](len=sample_num)}
        """
        re_dict = {'distance':self.distances, 'values': self.values}
        sample_list = []
        best_id, best_ssim = [0, 0], 0
        for sam_id in tqdm(range(self.num_sam), desc='Sample ', leave=False):
            code = self.codes[sam_id:sam_id + 1]
            for sem_id in tqdm(range(self.num_sem), desc='Semantic ', leave=False):
                result = defaultdict(list)
                boundary = self.boundaries[sem_id:sem_id + 1]
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
                    result['semantic_img_list'].append(image)
                    result['ssim'].append(ssim)
                if sum(result['ssim']) / len(self.distances) > best_ssim:
                    best_id = [sam_id, sem_id]
            re_dict[]
        re_dict.update({})
        return

    def save(self):
        prefix = (f'{cfg..model_name}_'
                  f'N{num_sam}_K{num_sem}_L{cfg..layer_idx}_seed{cfg..seed}')
        timestamp = datetime.datetime.now()
        version = '%d-%d-%d-%02.0d-%02.0d-%02.0d' % \
                  (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second)
        save_dir = os.path.join(cfg..save_dir, cfg..checkpoint_path.split('/')[-3],
                                f's{cfg..start_distance}e{cfg..end_distance}', version)
        os.makedirs(save_dir)
        vizer_1.save(os.path.join(save_dir, f'{prefix}_sample_first.html'))
        vizer_2.save(os.path.join(save_dir, f'{prefix}_semantic_first.html'))
