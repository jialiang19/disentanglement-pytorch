import torch.nn as nn
import torch 
from architectures.encoders.base.base_encoder import BaseImageEncoder
from common.ops import Flatten3D
from common.utils import init_layers


class SimpleConv64Block(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

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
            nn.Linear(256, latent_dim, bias=True)
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)


class ProConv64(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        self.columns = nn.ModuleList([])
        assert image_size == 64, 'This model only works with image size 64x64.'

    def forward(self, x):
        assert self.columns, 'ProConv64 should at least have one column (missing call to `new_task` ?)'
        for c in self.columns: 
            outputs = [c[0](x) for c in self.columns]
        if len(self.columns) == 1: 
            #print('----outputs[0].shape', outputs[0].shape) # [64, 1]
            return outputs[0], 0 
        outputs = torch.cat(outputs, dim=1)         
       # print('--------------------', len(self.columns)) 
        return outputs, 0 

    def new_task(self):
        task_id = len(self.columns)

        modules = []
        modules.append(SimpleConv64Block(self._latent_dim,
                                         self._num_channels, self._image_size))
        new_column = nn.ModuleList(modules)
        self.columns.append(new_column)

#        if self.use_cuda:
#            self.cuda()

    def freeze_columns(self, skip=None):
        if skip == None:
            skip = []

        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False



class ProGaussianConv64(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim * 2, num_channels, image_size)
        self.columns = nn.ModuleList([])
        assert image_size == 64, 'This model only works with image size 64x64.'

    def forward(self, x):
        assert self.columns, 'ProConv64 should at least have one column (missing call to `new_task` ?)'
        for c in self.columns: 
            outputs = [c[0](x) for c in self.columns]
        if len(self.columns) == 1: 
            #print('--- in encoder: outputs.shape',outputs[0].shape) # shape is [64, 4]  
            #print(outputs) 
            return outputs[0][:,0].unsqueeze(1), outputs[0][:,1].unsqueeze(1) # the first index is treated as mean, and the second index is treated as logvar  
        outputs = torch.cat(outputs, dim=1)         
        #print('--------------------', len(self.columns)) 
        #print(outputs.shape)  
        return outputs[:,0::2], outputs[:,1::2] # all the even indexes are treated as mean and all the odd indexes are treated as logvar  

    def new_task(self):
        task_id = len(self.columns)

        modules = []
        modules.append(SimpleConv64Block(self._latent_dim,
                                         self._num_channels, self._image_size))
        new_column = nn.ModuleList(modules)
        self.columns.append(new_column)

#        if self.use_cuda:
#            self.cuda()

    def freeze_columns(self, skip=None):
        if skip == None:
            skip = []

        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False


#class ProGaussianConv64(ProConv64):
#    def __init__(self, latent_dim, num_channels, image_size):
#        super().__init__(latent_dim * 2, num_channels, image_size)
#
#        # override value of _latent_dim
#        self._latent_dim = latent_dim
#
#    def forward(self, x):
#        mu_logvar = self.main(x)
#        mu = mu_logvar[:, :self._latent_dim]
#        logvar = mu_logvar[:, self._latent_dim:]
#        return mu, logvar
