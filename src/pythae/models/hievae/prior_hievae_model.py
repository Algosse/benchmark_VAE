import torch
import torch.nn.functional as F

from .hievae_config import HieVAEConfig

from pythae.models.nn.base_architectures import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.data.datasets import BaseDataset

from ..base import BaseAE

from .nn.utils import DmolNet

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
        y = y[:,:4,...].to(self.device)
        
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
        y = inputs["label"][:,:4,...].to(self.device)
        
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
        

