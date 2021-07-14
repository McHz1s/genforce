# python3.7
"""Collects all runners."""

from .stylegan_runner import StyleGANRunner
from .progan_runner import ProGANRunner
from .stylegan2_runner import StyleGAN2Runner

__all__ = ['StyleGANRunner', 'ProGANRunner', 'StyleGAN2Runner']
