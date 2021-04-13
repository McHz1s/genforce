import torch
import os
from copy import deepcopy
import json
from utils.misc import HtmlPageVisualizer, load_generator
from sefa.sefa import Sefa, SefaInfo, SefaSampleInfo
from datasets.datasets import MATLoader
import datetime


class SefaRunner(object):
    def __init__(self, configs, logger):
        self._name = self.__class__.__name__
        self.config = deepcopy(configs)
        self.logger = logger
        self.work_dir = self.config.work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.logger.info('Running Configuration:')
        config_str = json.dumps(self.config, indent=4).replace('"', '\'')
        self.logger.print(config_str + '\n')
        with open(os.path.join(self.work_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        self.generator = load_generator(configs.checkpoint_path, configs.generator_config)
        self.mat_loader = MATLoader(configs.gt_data_cfg.root_dir, configs.gt_data_cfg.degree_interval_list)
        self.mat_loader.fetch_img_mode = 'color'
        self.mat_loader.sort()
        # todo now step should be the same with gt img to conduct metircs
        if self.config.sefa_cfg.step > len(self.mat_loader):
            self.config.sefa_cfg.step = len(self.mat_loader)
        self.best_ssim, self.best_attampt, self.best_sample, self.best_semantic = 0, 0, 0, 0

    def sefa_cfg_generator(self):
        cfg = deepcopy(self.config.sefa_cfg)
        dis_start, dis_end, seed_range = cfg.start_distance, cfg.end_distance, cfg.seed_range
        for i, seed in enumerate(seed_range):
            start, end = dis_start, dis_end
            while start < end:
                yield seed, start, end
                start += 0.5
                end -= 0.5

    def run_step(self, sefa_cfg, attampt_id):
        sefa = Sefa(sefa_cfg, self.generator, self.mat_loader)
        output_sefa_info = sefa.inference()
        self.save_step(output_sefa_info, attampt_id, sefa_cfg)

    def save_step(self, sefa_info: SefaInfo, attampt_id, sefa_cfg):
        self.logger.info(f'Attampt {attampt_id}:')
        sam_id, sem_id, ssim = sefa_info.get_best_sample_and_semantic()
        self.logger.info(f'best_id: Sample_id:{sam_id}, Semantic_id: {sem_id}, SSIM: {ssim}')
        self.logger.info(f'Seed:{sefa_cfg.seed}, start_dis: {sefa_cfg.start_distance}, end_dis: {sefa_cfg.end_distance}')
        if self.best_ssim < ssim:
            self.best_ssim, self.best_attampt, self.best_sample, self.best_semantic = \
                ssim, attampt_id, sam_id, sem_id
        vizer = self.vis(sefa_info)
        save_path = os.path.join(self.config.work_dir, f'Attampt_{attampt_id}')
        os.makedirs(save_path)
        vizer.save(os.path.join(save_path, 'visulizer.html'))

    def run(self):
        for attampt_id, (seed, start, end) in enumerate(self.sefa_cfg_generator()):
            sefa_cfg = deepcopy(self.config.sefa_cfg)
            sefa_cfg.seed, sefa_cfg.start_distance, sefa_cfg.end_distance = seed, start, end
            self.run_step(sefa_cfg, attampt_id)
        self.logger.info('Finish')
        self.logger.info(f'Best: Attampt_id:{self.best_attampt} '
                         f'Sample_id:{self.best_sample}, '
                         f'Semantic_id: {self.best_semantic}, '
                         f'SSIM: {self.best_ssim}')

    def vis(self, sefa_info: SefaInfo):
        num_sam = len(sefa_info)
        num_sem = len(sefa_info[0])
        step = len(sefa_info[0][0]['img_list'])
        vizer = HtmlPageVisualizer(num_rows=num_sam * (num_sem + 1),
                                   num_cols=step + 1, viz_size=self.config.viz_size)
        for sam_id in range(num_sam):
            vizer.set_cell(sam_id * (num_sem + 1), 0,
                           text=f'Sample {sam_id:03d}',
                           highlight=True)
            for sem_id in range(num_sem):
                value = sefa_info.values[sem_id]
                vizer.set_cell(sam_id * (num_sem + 1) + sem_id + 1, 0,
                               text=f'Semantic {sem_id:03d}<br>({value:.3f})<br>SSIM:{sefa_info[sam_id][sem_id]["ssim"]}')
                for col_id, (img, ssim) in \
                        enumerate(zip(sefa_info[sam_id][sem_id]['img_list'], sefa_info[sam_id][sem_id]['ssim_list']),
                                  start=1):
                    vizer.set_cell(sam_id * (num_sem + 1) + sem_id + 1, col_id, image=img, text=f'{ssim:.4f}')
        return vizer

