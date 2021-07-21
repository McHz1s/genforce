# python3.7
"""Contains the runner for StyleGAN2."""

from copy import deepcopy

from .base_gan_runner import BaseGANRunner

__all__ = ['StyleGAN2Runner']


class StyleGAN2Runner(BaseGANRunner):
    """Defines the runner for StyleGAN2."""

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.lod = getattr(self, 'lod', None)

    def build_models(self):
        super().build_models()
        self.g_smooth_img = self.config.modules['generator'].get(
            'g_smooth_img', 10000)
        self.models['generator_smooth'] = deepcopy(self.models['generator'])

    def build_loss(self):
        super().build_loss()
        self.running_stats.add(
            f'Gs_beta', log_format='.4f', log_strategy='CURRENT')

    def train_step(self, data, **train_kwargs):
        # Update discriminator.
        self.set_model_requires_grad('discriminator', True)
        self.set_model_requires_grad('generator', False)

        d_loss = self.loss.d_loss(self, data)
        self.optimizers['discriminator'].zero_grad()
        d_loss.backward()
        self.optimizers['discriminator'].step()

        # Life-long update for generator.
        beta = 0.5 ** (self.batch_size * self.world_size / self.g_smooth_img)
        self.running_stats.update({'Gs_beta': beta})
        self.moving_average_model(model=self.models['generator'],
                                  avg_model=self.models['generator_smooth'],
                                  beta=beta)

        # Update generator.
        if self._iter % self.config.get('D_repeats', 1) == 0:
            self.set_model_requires_grad('discriminator', False)
            self.set_model_requires_grad('generator', True)
            g_loss = self.loss.g_loss(self, data)
            self.optimizers['generator'].zero_grad()
            g_loss.backward()
            self.optimizers['generator'].step()

