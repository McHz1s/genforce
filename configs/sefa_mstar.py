# python3.7
"""Configuration for training StyleGAN on FF-HQ (256) dataset.

All settings are particularly used for one replica (GPU), such as `batch_size`
and `num_workers`.
"""

viz_size = 128
gpus = '5'

work_dir = '/data3/lyz/cache/sefa'
checkpoint_path = '/data3/lyz/cache/genforce/stylegan_mstar_28z_degree1-2/2021-4-17-19-14-54/checkpoint_iter070000.pth'
gt_data_cfg = dict(root_dir='/data3/lyz/dataset/mstar/TRAINT72_132INF.MAT',
                   degree_interval_list=[[0, 90]])

generator_config = dict(
    gan_type='stylegan',
    resolution=128,
    z_space_dim=28, w_space_dim=28,
    image_channels=1, final_sigmoid=True
)

sefa_cfg = dict(num_samples=4, num_semantics=2, start_distance=-6, end_distance=6, step=118, seed_range=[0, 1, 2, 3],
                trunc_psi=1.0, trunc_layers=0, layer_idx='all')
