"""This module implements the base trainer allowing you to train the models implemented in pythae.

Available models:
------------------

.. autosummary::
    ~pythae.models.VDVAE
    :nosignatures:
"""

from .vdvae_trainer import VDVAETrainer
from .vdvae_trainer_config import VDVAETrainerConfig

__all__ = ["VDVAETrainer", "VDVAETrainerConfig"]
