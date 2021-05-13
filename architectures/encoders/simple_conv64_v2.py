import torch.nn as nn

from architectures.encoders.base.base_encoder import BaseImageEncoder
from common.ops import Flatten3D
from common.utils import init_layers


class SimpleConv64_v2(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'
        self.latent_dim = latent_dim 

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.ReLU(True),
            Flatten3D(),
        )
        self.fc = nn.Linear(256, latent_dim, bias=True) 
        init_layers(self._modules)

    def new_task(self): 
       # print('fc_weight_shape:', self.fc.weight.shape) 
       # print('fc_bias_shape:', self.fc.bias.shape)
       # print('fc_bias_value_old:', self.fc.bias)
       # print('fc_weight_value_old:', self.fc.weight.data[0][1],  self.fc.weight.data[0][10], self.fc.weight.data[0][25])

        old_bias = self.fc.bias
        old_weight = self.fc.weight
        self.latent_dim += 1 
        self.fc = nn.Linear(256, self.latent_dim, bias=True)  
      
       #  print('fc_weight_shape:', self.fc.weight.shape) 
       #  print('fc_bias_shape:', self.fc.bias.shape)
       #  print('fc_bias_value_new:', self.fc.bias)  
       #  print('fc_weight_value_new:', self.fc.weight.data[0][1],  self.fc.weight.data[0][10], self.fc.weight.data[0][25])

        self.fc.bias.data[0:-1] = old_bias.data[0:]  
        self.fc.weight.data[0:-1] = old_weight.data[0:]  
        
       #  print('fc_weight_value_replace:', self.fc.weight.data[0][1],  self.fc.weight.data[0][10], self.fc.weight.data[0][25])
       #  print('fc_bias_value_replaced:', self.fc.bias)  
    
    def forward(self, x):
        return self.fc(self.main(x)), 0 


class SimpleGaussianConv64(SimpleConv64_v2):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim * 2, num_channels, image_size)

        # override value of _latent_dim
        self._latent_dim = latent_dim

    def forward(self, x):
        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self._latent_dim]
        logvar = mu_logvar[:, self._latent_dim:]
        return mu, logvar
