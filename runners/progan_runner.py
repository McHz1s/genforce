# python3.7
"""Contains the runner for ProGAN."""

from copy import deepcopy

from .stylegan_runner import StyleGANRunner

__all__ = ['ProGANRunner']


class ProGANRunner(StyleGANRunner):
    """Defines the runner for StyleGAN."""

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.lod = getattr(self, 'lod', None)

    def train_step(self, data, **train_kwargs):
        # Set level-of-details.
        G = self.get_module(self.models['generator'])
        D = self.get_module(self.models['discriminator'])
        Gs = self.get_module(self.models['generator_smooth'])
        G.lod.data.fill_(self.lod)
        D.lod.data.fill_(self.lod)
        Gs.lod.data.fill_(self.lod)

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
