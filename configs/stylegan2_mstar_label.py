# python3.7
"""Configuration for training StyleGAN on FF-HQ (256) dataset.

All settings are particularly used for one replica (GPU), such as `batch_size`
and `num_workers`.
"""

runner_type = 'StyleGAN2Runner'
gan_type = 'stylegan2'
resolution = 128
batch_size = 8
val_batch_size = 32
total_img = 25000_000
gpus = '7'

# Training dataset is repeated at the beginning to avoid loading dataset
# repeatedly at the end of each epoch. This can save some I/O time.
data = dict(
    num_workers=0,
    repeat=500,
    train=dict(root_dir='/home/lyz/dataset/MSTAR/TRAINT72_132INF.MAT', data_format='MAT',
               resolution=resolution, transform=None, degree_interval_list=[[0, 90]], label_size=90),
    val=dict(root_dir='/home/lyz/dataset/MSTAR/TRAINT72_132INF.MAT', data_format='MAT',
             resolution=resolution, run_mode='metric', transform=None, degree_interval_list=[[0, 90]], label_size=90),
)

controllers = dict(
    RunningLogger=dict(every_n_iters=10),
    ProgressScheduler=dict(
        every_n_iters=1, init_res=128, minibatch_repeats=4,
        lod_training_img=10000, lod_transition_img=10000,
        batch_size_schedule=dict(res128=8),
    ),
    Snapshoter=dict(every_n_iters=500, first_iter=True, num=200),
    FIDEvaluator=dict(every_n_iters=5000, first_iter=True, num=50000),
    Checkpointer=dict(every_n_iters=2500, first_iter=True),
)

modules = dict(
    discriminator=dict(
        model=dict(gan_type=gan_type, resolution=resolution, image_channels=1, label_size=90),
        lr=dict(lr_type='FIXED'),
        opt=dict(opt_type='Adam', base_lr=1e-3, betas=(0.0, 0.99)),
        kwargs_train=dict(),
        kwargs_val=dict(),
    ),
    generator=dict(
        model=dict(gan_type=gan_type, resolution=resolution, image_channels=1, final_sigmoid=True,
                   z_space_dim=32, w_space_dim=32, label_size=90),
        lr=dict(lr_type='FIXED'),
        opt=dict(opt_type='Adam', base_lr=1e-3, betas=(0.0, 0.99)),
        kwargs_train=dict(w_moving_decay=0.995, style_mixing_prob=0.9, trunc_psi=1.0, trunc_layers=0, randomize_noise=False),
        kwargs_val=dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False),
        g_smooth_img=10_000,
    )
)

loss = dict(
    type='LogisticGANLoss',
    d_loss_kwargs=dict(r1_gamma=10.0),
    g_loss_kwargs=dict(),
)
