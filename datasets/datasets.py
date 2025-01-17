# python3.7
"""Contains the class of dataset."""

import os
import pickle
import string
import zipfile
import numpy as np
import cv2
import lmdb

import torch
from torch.utils.data import Dataset

import scipy.io as scio
import copy
from collections import defaultdict
from utils.visualizer import plt_show

from .transforms import progressive_resize_image
from .transforms import crop_resize_image
from .transforms import resize_image
from .transforms import normalize_image

__all__ = ['BaseDataset']

_FORMATS_ALLOWED = ['dir', 'lmdb', 'list', 'zip', 'mat']

class ZipLoader(object):
    """Defines a class to load zip file.

    This is a static class, which is used to solve the problem that different
    data workers can not share the same memory.
    """
    files = dict()

    @staticmethod
    def get_zipfile(file_path):
        """Fetches a zip file."""
        zip_files = ZipLoader.files
        if file_path not in zip_files:
            zip_files[file_path] = zipfile.ZipFile(file_path, 'r')
        return zip_files[file_path]

    @staticmethod
    def get_image(file_path, image_path):
        """Decodes an image from a particular zip file."""
        zip_file = ZipLoader.get_zipfile(file_path)
        image_str = zip_file.read(image_path)
        image_np = np.frombuffer(image_str, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image


def loadMSTAR(data_dir, degree_left=0, degree_right=180):
    data = scio.loadmat(data_dir)
    imadata = data['imadata']
    imalabel = np.squeeze(data['AZ'])
    imadata = np.transpose(imadata, [2, 0, 1])
    imadata = np.expand_dims(imadata, axis=1)
    images = []
    for i in range(imadata.shape[0]):
        I = abs(imadata[i, :, :, :])
        I = (I - np.min(np.min(I))) / (np.max(np.max(I)) - np.min(np.min(I)))
        I *= 255.
        if degree_left <= imalabel[i] <= degree_right:
            images.append(I)
    images = np.array(images)
    images = images.astype(np.float32)
    print(images.shape)
    return images


class MATLoader(object):
    """Defines a class to load mat file.
    """

    def __init__(self, file_path, degree_interval_list=[[0, 180]]):
        self.raw_data_dict = scio.loadmat(file_path)
        self.post_data_dict = self.process(degree_interval_list)
        self.fetch_img_mode = 'gray'

    def process(self, degree_interval_list):
        """
        degree_list: set{list[]}
        example:
            [{][0,18],[90,120]]: keep mstar image degree in [0,18] or [90,120]
        """
        post_data_dict = defaultdict(list)
        for i, degree in enumerate(self.raw_data_dict['AZ'][0]):
            if any(list(map(lambda interval: interval[1] >= degree >= interval[0], degree_interval_list))):
                for key, value in self.raw_data_dict.items():
                    if key in ['__header__', '__version__', '__globals__']:
                        continue
                    if key == 'imadata':
                        x = value[:, :, i]
                        x = abs(x)
                        x = (x - np.min(np.min(x))) / (np.max(np.max(x)) - np.min(np.min(x)))
                        x = np.expand_dims(x, -1)
                        color_img = cv2.cvtColor((255 * x).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                        post_data_dict['gray_imadata'].append(x)
                        post_data_dict['color_imadata'].append(color_img)
                    else:
                        x = value[0, i]
                        post_data_dict[key].append(x)
        return post_data_dict

    def sort(self, reverse=False):
        for key, item in self.post_data_dict.items():
            if key == 'AZ':
                continue
            gather = list(zip(item, self.post_data_dict['AZ']))
            gather.sort(key=lambda x: x[1], reverse=reverse)
            self.post_data_dict[key] = [x for x, _ in gather]
        self.post_data_dict['AZ'].sort(reverse=reverse)

    def get_post_mat_data(self):
        return self.post_data_dict

    def __len__(self):
        return len(self.post_data_dict[f'{self.fetch_img_mode}_imadata'])

    def __getitem__(self, idx):
        return self.post_data_dict[f'{self.fetch_img_mode}_imadata'][idx]


class LmdbLoader(object):
    """Defines a class to load lmdb file.

    This is a static class, which is used to solve lmdb loading error
    when num_workers > 0
    """
    files = dict()

    @staticmethod
    def get_lmdbfile(file_path):
        """Fetches a lmdb file"""
        lmdb_files = LmdbLoader.files
        if 'env' not in lmdb_files:
            env = lmdb.open(file_path,
                            max_readers=1,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            with env.begin(write=False) as txn:
                num_samples = txn.stat()['entries']
            cache_file = '_cache_' + ''.join(
                c for c in file_path if c in string.ascii_letters)
            if os.path.isfile(cache_file):
                keys = pickle.load(open(cache_file, "rb"))
            else:
                with env.begin(write=False) as txn:
                    keys = [key for key, _ in txn.cursor()]
                pickle.dump(keys, open(cache_file, "wb"))
            lmdb_files['env'] = env
            lmdb_files['num_samples'] = num_samples
            lmdb_files['keys'] = keys
        return lmdb_files

    @staticmethod
    def get_image(file_path, idx):
        """Decodes an image from a particular lmdb file"""
        lmdb_files = LmdbLoader.get_lmdbfile(file_path)
        env = lmdb_files['env']
        keys = lmdb_files['keys']
        with env.begin(write=False) as txn:
            imagebuf = txn.get(keys[idx])
        image_np = np.frombuffer(imagebuf, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image


class BaseDataset(Dataset):
    """Defines the base dataset class.

    This class supports loading data from a full-of-image folder, a lmdb
    database, or an image list. Images will be pre-processed based on the given
    `transform` function before fed into the data loader.

    NOTE: The loaded data will be returned as a directory, where there must be
    a key `image`.
    """

    def __init__(self,
                 root_dir,
                 resolution,
                 data_format='dir',
                 image_list_path=None,
                 mirror=0.0,
                 progressive_resize=True,
                 crop_resize_resolution=-1,
                 transform=normalize_image,
                 transform_kwargs=None,
                 run_mode='train',
                 degree_interval_list=[[0, 180]],
                 **_unused_kwargs):
        """Initializes the dataset.

        Args:
            root_dir: Root directory containing the dataset.
            resolution: The resolution of the returned image.
            data_format: Format the dataset is stored. Supports `dir`, `lmdb`,
                and `list`. (default: `dir`)
            image_list_path: Path to the image list. This field is required if
                `data_format` is `list`. (default: None)
            mirror: The probability to do mirror augmentation. (default: 0.0)
            progressive_resize: Whether to resize images progressively.
                (default: True)
            crop_resize_resolution: The resolution of the output after crop
                and resize. (default: -1)
            transform: The transform function for pre-processing.
                (default: `datasets.transforms.normalize_image()`)
            transform_kwargs: The additional arguments for the `transform`
                function. (default: None)

        Raises:
            ValueError: If the input `data_format` is not supported.
            NotImplementedError: If the input `data_format` is not implemented.
        """
        if data_format.lower() not in _FORMATS_ALLOWED:
            raise ValueError(f'Invalid data format `{data_format}`!\n'
                             f'Supported formats: {_FORMATS_ALLOWED}.')

        self.root_dir = root_dir
        self.resolution = resolution
        self.data_format = data_format.lower()
        self.image_list_path = image_list_path
        self.mirror = np.clip(mirror, 0.0, 1.0)
        self.progressive_resize = progressive_resize
        self.crop_resize_resolution = crop_resize_resolution
        self.transform = transform
        self.transform_kwargs = transform_kwargs or dict()
        self.run_mode = run_mode

        if self.data_format == 'dir':
            self.image_paths = sorted(os.listdir(self.root_dir))
            self.num_samples = len(self.image_paths)
        elif self.data_format == 'lmdb':
            lmdb_file = LmdbLoader.get_lmdbfile(self.root_dir)
            self.num_samples = lmdb_file['num_samples']
        elif self.data_format == 'list':
            self.metas = []
            assert os.path.isfile(self.image_list_path)
            with open(self.image_list_path) as f:
                for line in f:
                    fields = line.rstrip().split(' ')
                    if len(fields) == 1:
                        self.metas.append((fields[0], None))
                    else:
                        assert len(fields) == 2
                        self.metas.append((fields[0], int(fields[1])))
            self.num_samples = len(self.metas)
        elif self.data_format == 'zip':
            zip_file = ZipLoader.get_zipfile(self.root_dir)
            image_paths = [f for f in zip_file.namelist()
                           if ('.jpg' in f or '.jpeg' in f or '.png' in f)]
            self.image_paths = sorted(image_paths)
            self.num_samples = len(self.image_paths)
        elif self.data_format == 'mat':
            self.mat_loader = MATLoader(self.root_dir, degree_interval_list=degree_interval_list)
            self.num_samples = len(self.mat_loader)
        else:
            raise NotImplementedError(f'Not implemented data format '
                                      f'`{self.data_format}`!')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = dict()

        # Load data.
        if self.data_format == 'dir':
            # TODO: support turbo-jpeg backend
            image_path = self.image_paths[idx]
            image = cv2.imread(os.path.join(self.root_dir, image_path))
        elif self.data_format == 'lmdb':
            image = LmdbLoader.get_image(self.root_dir, idx)
        elif self.data_format == 'list':
            image_path, label = self.metas[idx]
            image = cv2.imread(os.path.join(self.root_dir, image_path))
            label = None if label is None else torch.LongTensor(label)
            data.update({'label': label})
        elif self.data_format == 'zip':
            image_path = self.image_paths[idx]
            image = ZipLoader.get_image(self.root_dir, image_path)
        elif self.data_format == 'mat':
            image = self.mat_loader[idx]
        else:
            raise NotImplementedError(f'Not implemented data format '
                                      f'`{self.data_format}`!')

        image = image[:, :, ::-1]  # Converts BGR (cv2) to RGB.
        if self.run_mode == 'metric':
            cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # Transform image.
        if self.crop_resize_resolution > 0:
            image = crop_resize_image(image, self.crop_resize_resolution)
        if self.progressive_resize:
            image = progressive_resize_image(image, self.resolution)
        image = image.transpose(2, 0, 1).astype(np.float32)
        if np.random.uniform() < self.mirror:
            image = image[:, :, ::-1]  # CHW
        image = torch.FloatTensor(image.copy())
        if not self.progressive_resize:
            image = resize_image(image, self.resolution)

        if self.transform is not None:
            image = self.transform(image, **self.transform_kwargs)
        data.update({'image': image})

        return data

