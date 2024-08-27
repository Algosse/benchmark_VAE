from pythae.trainers import BaseTrainer
from pythae.models import BaseAE
from pythae.data.datasets import BaseDataset

from typing import Optional, List

from .vdvae_trainer_config import VDVAETrainerConfig

from pythae.trainers.training_callbacks import TrainingCallback

class VDVAETrainer(BaseTrainer):
    """
        Trainer designed to work with VDVAE models.
    """
    
    def __init__(
        self,
        model: BaseAE,
        train_dataset: BaseDataset,
        eval_dataset: Optional[BaseDataset] = None,
        training_config: Optional[VDVAETrainerConfig] = None,
        callbacks: List[TrainingCallback] = None,
    ):
        super().__init__(model, train_dataset, eval_dataset, training_config, callbacks)
        
    def predict(self, model: BaseAE):
        """
        VDVAE implements multiple latent spaces. The logic is already implemented in the sample method of the model.
        """
        
        model.eval()

        inputs = next(iter(self.eval_loader))
        inputs = self._set_inputs_to_device(inputs)
        
        model_out = model(inputs)
        
        reconstructions = model_out.recon_x.cpu().detach()[
            : min(inputs['data'].shape[0], 10), :inputs['data'].shape[1]
        ]
        
        # Here, we can not simply draw z and use it for the decoder as we have multiple latent spaces.
        if model.model_config.is_conditioned:
            if self.distributed:
                generation = model.module.sample(y=inputs['label'][:min(inputs['data'].shape[0], 10)]).recon_x.cpu().detach()
            else:
                generation = model.sample(y=inputs['label'][:min(inputs['data'].shape[0], 10)]).recon_x.cpu().detach()
        else:
            if self.distributed:
                generation = model.module.sample(n=min(inputs['data'].shape[0], 10)).recon_x.cpu().detach()
            else:
                generation = model.sample(n=min(inputs['data'].shape[0], 10)).recon_x.cpu().detach()
        
        return (
            inputs['data'][: min(inputs['data'].shape[0], 10)],
            reconstructions,
            generation,
        )