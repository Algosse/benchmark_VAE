from pythae.models.hievae.nn.decoder import HieVAEDecoder
from pythae.models.hievae.nn.swin_transformer_config import SwinTransformerConfig
from pythae.models.hievae.nn.swin_transformer_encoder import SequenceSwinTransformer
import torch
import torch.nn.functional as F

from .hievae_config import HieVAEConfig

from pythae.models.nn.base_architectures import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.data.datasets import BaseDataset

from ..base import BaseAE

from .nn.utils import DmolNet

import os
from ..base.base_utils import (
    CPU_Unpickler,
)

class PriorHieVAE(BaseAE):
    
    def __init__(self, model_config: HieVAEConfig, encoder: BaseEncoder, decoder: BaseDecoder, prior: BaseEncoder):
        BaseAE.__init__(self, model_config=model_config, decoder=decoder)
        self.model_name = "PriorHieVAE"
        
        self.model_config = model_config

        self.set_encoder(encoder)
        
        self.prior = prior
        
        if model_config.reconstruction_loss == "dmol":
            self.out_net = DmolNet(model_config.features[0], model_config.num_mixtures)
    
    def predict(self, y, **kwargs):
        y = y.to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.model_config.mixed_precision):
            prior_activations = self.prior(y)['activations']
            out = self.decoder(None, prior_activations)

        if self.model_config.reconstruction_loss == "dmol":
            recon_x = self.out_net.sample(out['recon_dist'])
        else:
            recon_x = out['recon_dist'][:, :self.model_config.input_dim[0], ...]
        
        output = ModelOutput(
            recon_x = recon_x,
        )
        
        return output

    def forward(self, inputs: BaseDataset, **kwargs):
        
        x = inputs["data"].to(self.device)
        y = inputs["label"].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.model_config.mixed_precision):
            activations = self.encoder(y, x)['activations']
            prior_activations = self.prior(y)['activations']
            
            out = self.decoder(activations, prior_activations)
            
            if self.model_config.reconstruction_loss == "dmol":
                recon_x = self.out_net.sample(out['recon_dist'])
            else:
                recon_x = out['recon_dist'][:, :self.model_config.input_dim[0], ...]
            
            recon_dist = out['recon_dist']
            kl = out['kl']
            
            loss, distortion, rate = self.loss_function(recon_dist, x, kl)
        
        output = ModelOutput(
            recon_loss=distortion,
            reg_loss=rate,
            loss=loss,
            recon_x=recon_x,
            recon_dist=recon_dist,
        )
        
        return output

    def loss_function(self, recon_x, x, kls):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            )

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            )
        
        elif self.model_config.reconstruction_loss == "normal":
            mean = recon_x[:, :self.model_config.input_dim[0], ...]
            std = recon_x[:, self.model_config.input_dim[0]:, ...]
            dist = torch.distributions.Normal(mean.reshape(x.shape[0], -1), torch.clamp(std.reshape(x.shape[0], -1), min=0.01))
            recon_loss = -dist.log_prob(x.reshape(x.shape[0], -1))

        elif self.model_config.reconstruction_loss == "dmol":
            recon_loss = self.out_net.nll(recon_x, x).reshape(x.shape[0], -1)
        
        distortion_per_pixel = recon_loss.mean(dim=-1)
        
        rate_per_pixel = 0
        for kl in kls:
            rate_per_pixel += kl.mean(dim=(1, 2, 3))
        rate_per_pixel /= len(kls)

        return (distortion_per_pixel + rate_per_pixel).mean(dim=0), distortion_per_pixel.mean(dim=0), rate_per_pixel.mean(dim=0)
        

    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """

        model_config = HieVAEConfig.from_json_file(os.path.join(dir_path, 'model_config.json'))

        # drop qk_scale from encoder_config dict
        encoder_config = model_config.encoder_config

        if 'qk_scale' in encoder_config and encoder_config['qk_scale'] == None:
            # if qk_scale is null, drop it because typing does not like Optional[float] = None
            encoder_config.pop('qk_scale')
        
        prior_config = SwinTransformerConfig.from_dict(encoder_config)
        
        encoder_config = SwinTransformerConfig(**prior_config.to_dict())
        encoder_config.sequence_size += 1


        encoder = SequenceSwinTransformer(model_config, encoder_config)
        prior = SequenceSwinTransformer(model_config, prior_config)
        decoder = HieVAEDecoder(model_config)
        
        model = PriorHieVAE(model_config, encoder, decoder, prior)
        
        state_dict = torch.load(os.path.join(dir_path, 'model.pt'))
        model.load_state_dict(state_dict['model_state_dict'])

        return model
    
