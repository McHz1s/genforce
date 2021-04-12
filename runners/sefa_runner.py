import torch
import os
from copy import deepcopy
import json
from utils.misc import HtmlPageVisualizer, load_generator
from sefa.sefa import Sefa


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
        self.generator = load_generator(configs.checkpoint_path, configs.model_name)

    def sefa_cfg_generator(self):
        cfg = deepcopy(self.config.sefa_cfg)
        distance_range, seed_range = cfg.distance_range, cfg.seed_range
        for i, seed in enumerate(seed_range):
            start, end = distance_range
            while start < end:
                yield seed, start, end
                start += 0.5
                end -= 0.5

    def run_step(self, sefa_cfg):
        sefa = Sefa(sefa_cfg, self.generator, self.gt_img)
        output_dict = sefa.inference()

    def run(self):
        for seed, start, end in self.sefa_cfg_generator():
            sefa_cfg = deepcopy(self.config.sefa_cfg)
            sefa_cfg.seed, sefa_cfg.start_distance, sefa_cfg.end_distance = seed, start, end
            self.run_step(sefa_cfg)

    def vis(self, sefa_info):
        num_sam = len(sefa_info)
        num_sem = len(sefa_info['semantic_list'])
        step = len(sefa_info_list[0]['semantic_list'][0])
        vizer = HtmlPageVisualizer(num_rows=num_sam * (num_sem + 1), num_cols=step + 1, viz_size=self.config.viz_size)
        for sem_id in range(num_sem):
            value = sefa_info['values'][sem_id]
            vizer.set_cell(sem_id, 0,
                           text=f'Semantic {sem_id:03d}<br>({value:.3f})<br>SSIM:{sefa_info["SSIM_list"][sem_id]}',
                           highlight=True)

        for sam_id in tqdm(range(num_sam), desc='Sample ', leave=False):
            code = codes[sam_id:sam_id + 1]
            for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
                boundary = boundaries[sem_id:sem_id + 1]
                for col_id, d in enumerate(distances, start=1):
                    temp_code = code.copy()
                    if gan_type == 'pggan':
                        temp_code += boundary * d
                        image = generator(to_tensor(temp_code))['image']
                    elif gan_type in ['stylegan', 'stylegan2']:
                        temp_code[:, layers, :] += boundary * d
                        image = generator.synthesis(to_tensor(temp_code))['image']
                    image = post_process(image, transpose=True)[0]
                    vizer_1.set_cell(sem_id * (num_sam + 1) + sam_id + 1, col_id,
                                     image=image)
                    vizer_2.set_cell(sam_id * (num_sem + 1) + sem_id + 1, col_id,
                                     image=image)

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
