import logging

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from models.base.base_disentangler import BaseDisentangler
from architectures import encoders, decoders
from common.utils import get_scheduler
from torchsummary import summary


class PROAEModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        z, _ = self.encode(x)
        return self.decode(z)


class PROAE(BaseDisentangler):
    def __init__(self, args):
        super().__init__(args)

        # encoder and decoder
        self.encoder_name = args.encoder[0]
        self.decoder_name = args.decoder[0]
        encoder = getattr(encoders, self.encoder_name)
        decoder = getattr(decoders, self.decoder_name)
        self.args = args

        # model and optimizer
        new_encoder = encoder(self.z_dim, self.num_channels,
                              self.image_size).to(self.device)
        new_encoder.new_task()
        self.model = PROAEModel(new_encoder,
                                decoder(self.z_dim, self.num_channels, self.image_size)).to(self.device)
        self.optim_G = optim.Adam(self.model.parameters(
        ), lr=self.lr_G, betas=(self.beta1, self.beta2))

        # nets
        self.nets = [self.model]
        self.net_dict = {
            'G': self.model
        }
        self.optim_dict = {
            'optim_G': self.optim_G,
        }

        # to do: figure out how to do with setup_scheduler later
        self.setup_schedulers(args.lr_scheduler, args.lr_scheduler_args,
                              args.w_recon_scheduler, args.w_recon_scheduler_args)

    def loss_fn(self, **kwargs):
        x_recon = kwargs['x_recon']
        x_true = kwargs['x_true']
        bs = self.batch_size
        recon_loss = F.binary_cross_entropy(
            x_recon, x_true, reduction='sum') / bs * self.w_recon

        return recon_loss

    def increase_dim(self):
        print('increase the latent dimension by 1')
        self.model.encoder.freeze_columns()
        self.model.encoder.new_task()
        decoder = getattr(decoders, self.decoder_name)
        new_decoder = decoder(len(self.model.encoder.columns),
                              self.num_channels, self.image_size).to(self.device)
        self.model.decoder = new_decoder
        self.optim_G = optim.Adam(self.model.parameters(
        ), lr=self.lr_G, betas=(self.beta1, self.beta2))
        # nets
        self.nets = [self.model]
        self.net_dict['G'] = self.model
        self.optim_dict['optim_G'] = self.optim_G

        # To Do: figure out what to do with scheduler later
        self.setup_schedulers(self.args.lr_scheduler, self.args.lr_scheduler_args,
                              self.args.w_recon_scheduler, self.args.w_recon_scheduler_args)
        self.model.to(self.device)

    def train(self):
        while not self.training_complete():
            self.net_mode(train=True)
            for x_true1, _ in self.data_loader:
                x_true1 = x_true1.to(self.device)
                x_recon = self.model(x_true1)

                recon_loss = self.loss_fn(x_recon=x_recon, x_true=x_true1)
                loss_dict = {'recon': recon_loss}

                self.optim_G.zero_grad()
                recon_loss.backward(retain_graph=True)
                self.optim_G.step()

                self.log_save(loss=loss_dict,
                              input_image=x_true1,
                              recon_image=x_recon,
                              )

                if self.iter >= int(self.num_batches * self.args.inc_every_n_epoch) and self.iter % int(self.num_batches * self.args.inc_every_n_epoch) == 0:
                    self.increase_dim()
                    #summary(self.model, (1, 64, 64))
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            print('required', name,param.data.shape) 
                        else: 
                            print('not-required', name,param.data.shape)
           # end of epoch
        self.pbar.close()

    def test(self):
        self.net_mode(train=False)
        for x_true1, _ in self.data_loader:
            x_true1 = x_true1.to(self.device)
            x_recon = self.model(x_true1)

            self.visualize_recon(x_true1, x_recon, test=True)
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max), spacing=self.traverse_spacing,
                                    data=(x_true1, None), test=True)

            self.iter += 1
            self.pbar.update(1)
