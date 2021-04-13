# python3.7
"""Configuration for training StyleGAN on FF-HQ (256) dataset.

All settings are particularly used for one replica (GPU), such as `batch_size`
and `num_workers`.
"""

runner_type = 'StyleGANRunner'
gan_type = 'stylegan'
resolution = 128
batch_size = 24
val_batch_size = 64
total_img = 25000_000
gpus = '6'

# Training dataset is repeated at the beginning to avoid loading dataset
# repeatedly at the end of each epoch. This can save some I/O time.
data = dict(
    num_workers=4,
    repeat=500,
    train=dict(root_dir='/data3/lyz/dataset/mstar/TRAINT72_132INF.MAT', data_format='MAT',
               resolution=resolution, transform=None, degree_interval_list=[[10*x + 5, 10*x + 10] for x in range(9, 18)]),
    val=dict(root_dir='/data3/lyz/dataset/mstar/TRAINT72_132INF.MAT', data_format='MAT',
             resolution=resolution, run_mode='metric', degree_interval_list=[[90, 180]], transform=None),
)

controllers = dict(
    RunningLogger=dict(every_n_iters=10),
    ProgressScheduler=dict(
        every_n_iters=1, init_res=16, minibatch_repeats=4,
        lod_training_img=150000, lod_transition_img=150000,
        batch_size_schedule=dict(res4=64, res8=128, res16=128, res32=64),
    ),
    Snapshoter=dict(every_n_iters=500, first_iter=True, num=200),
    FIDEvaluator=dict(every_n_iters=5000, first_iter=True, num=50000),
    Checkpointer=dict(every_n_iters=10000, first_iter=True),
)

modules = dict(
    discriminator=dict(
        model=dict(gan_type=gan_type, resolution=resolution, image_channels=1),
        lr=dict(lr_type='FIXED'),
        opt=dict(opt_type='Adam', base_lr=1e-3, betas=(0.0, 0.99)),
        kwargs_train=dict(),
        kwargs_val=dict(),
    ),
    generator=dict(
        model=dict(gan_type=gan_type, resolution=resolution, z_space_dim=256, w_space_dim=256, image_channels=1, final_sigmoid=True),
        lr=dict(lr_type='FIXED'),
        opt=dict(opt_type='Adam', base_lr=1e-3, betas=(0.0, 0.99)),
        kwargs_train=dict(w_moving_decay=0.995, style_mixing_prob=0.9, trunc_psi=1.0, trunc_layers=0),
        kwargs_val=dict(trunc_psi=1.0, trunc_layers=0, randomize_noise=False),
        g_smooth_img=10_000,
    )
)

loss = dict(
    type='LogisticGANLoss',
    d_loss_kwargs=dict(r1_gamma=10.0),
    g_loss_kwargs=dict(),
)
