from datasets.datasets import BaseDataset
from torch.utils.data import DataLoader

import cv2
import os

if __name__ == '__main__':
    dataset = BaseDataset(root_dir='/data3/lyz/dataset/mstar/TRAINT72_132INF.MAT', data_format='MAT',
                          resolution=128, transform=None, degree_interval_list=[[2*x +1, 2*x+2] for x in range(0, 180)])
    for degree, img in zip(dataset.mat_loader.post_data_dict['AZ'],
                           dataset.mat_loader.post_data_dict['color_imadata']):
        name = os.path.join('/data3/lyz/mstar_color_90', f'{int(degree):03d}.png')
        cv2.imwrite(name, img)
