import torch

from pythae.trainers import BaseTrainer
from pythae.models import BaseAE
from pythae.data.datasets import BaseDataset

from typing import Optional, List

from pythae.trainers import BaseTrainerConfig
from pythae.trainers.training_callbacks import TrainingCallback

class HieVAETrainer(BaseTrainer):
    
    def __init__(
        self,
        model: BaseAE,
        train_dataset: BaseDataset,
        eval_dataset: Optional[BaseDataset] = None,
        training_config: Optional[BaseTrainerConfig] = None,
        callbacks: List[TrainingCallback] = None,
    ):
        super().__init__(model, train_dataset, eval_dataset, training_config, callbacks)
        
    def predict(self, model: BaseAE):
        
        model.eval()

        inputs = next(iter(self.eval_loader))
        inputs = self._set_inputs_to_device(inputs)
        
        model_out = model(inputs)
        
        reconstructions = model_out.recon_x.cpu().detach()[
            : min(inputs['data'].shape[0], 10), :inputs['data'].shape[1]
        ]
        
        generation = model.predict(inputs['label']).recon_x.cpu().detach()[
            : min(inputs['data'].shape[0], 10), :inputs['data'].shape[1]
        ]
        
        return (
            inputs['data'][: min(inputs['data'].shape[0], 10)],
            reconstructions,
            generation,
        )