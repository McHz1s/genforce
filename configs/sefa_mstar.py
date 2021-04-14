# python3.7
"""Configuration for training StyleGAN on FF-HQ (256) dataset.

All settings are particularly used for one replica (GPU), such as `batch_size`
and `num_workers`.
"""

viz_size = 128
gpus = '8'

work_dir = '/data3/lyz/cache/sefa'
checkpoint_path = '/data3/lyz/cache/genforce/stylegan_mstar_8z_degree5-10/2021-4-14-01-31-29/checkpoint_iter050000.pth'
gt_data_cfg = dict(root_dir='/data3/lyz/dataset/mstar/TRAINT72_132INF.MAT',
                   degree_interval_list=[[0, 90]])

generator_config = dict(
    gan_type='stylegan',
    resolution=128,
    z_space_dim=8, w_space_dim=8,
    image_channels=1, final_sigmoid=True
)

sefa_cfg = dict(num_samples=2, num_semantics=3, start_distance=-7, end_distance=7, step=118, seed_range=[0, 1],
                trunc_psi=1.0, trunc_layers=0, layer_idx='all')
