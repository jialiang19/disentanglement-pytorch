import torch.nn as nn

from architectures.encoders.base.base_encoder import BaseImageEncoder
from common.ops import Unsqueeze3D
from common.utils import init_layers


class SimpleConv64_v2(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'
        
        self.latent_dim = latent_dim 
        self.main = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, num_channels, 3, 1)
        )
        # output shape = bs x 3 x 64 x 64

        self.unsqueeze = Unsqueeze3D()
        self.first_conv= nn.Conv2d(latent_dim, 256, 1, 2)
        init_layers(self._modules)
    
    def new_task(self): 
        self.latent_dim += 1 
        old_weight = self.first_conv.weight 
        self.first_conv= nn.Conv2d(self.latent_dim, 256, 1, 2) 
        self.first_conv.weight.data[:,:-1,:,:] = old_weight.data[:,:,:,:] 
    
    def forward(self, x):
        #print('x_shape_1:',x.shape) 
        x = self.unsqueeze(x) 
        #print('x_shape_2:',x.shape)  
        x = self.first_conv(x) 
        #print('x_shape_3:',x.shape)  
        return self.main(x)
