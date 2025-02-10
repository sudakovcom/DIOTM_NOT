import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


class ToyDiscriminator(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128):
        super().__init__()
        self.out_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.x_module = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.t_module = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, t, x):
        t = get_timestep_embedding(t, 128)
        temb = self.t_module(t)
        xemb = self.x_module(x)
        return self.out_module(xemb + temb)


class ToyGenerator(torch.nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128, z_dim=2, zero_out_last_layer=False):
        super(ToyGenerator, self).__init__()
        self.zero_out_last_layer = zero_out_last_layer
        hid = hidden_dim

        self.z_module = nn.Sequential(
            nn.Linear(z_dim, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
        )

        self.x_module = ResNet_FC(data_dim, hidden_dim, num_res_blocks=3)

        self.out_module = nn.Sequential(
            nn.Linear(hid, hid),
            nn.SiLU(),
            nn.Linear(hid, data_dim),
        )
        if zero_out_last_layer:
            self.out_module[-1] = zero_module(self.out_module[-1])

    def forward(self, x, z):
        z_out = self.z_module(z)
        x_out = self.x_module(x)
        out   = self.out_module(x_out+z_out)

        return out


class ResNet_FC(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_res_blocks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.map=nn.Linear(data_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [self.build_res_block() for _ in range(num_res_blocks)])

    def build_linear(self, in_features, out_features):
        linear = nn.Linear(in_features, out_features)
        return linear

    def build_res_block(self):
        hid = self.hidden_dim
        layers = []
        widths =[hid]*4
        for i in range(len(widths) - 1):
            layers.append(self.build_linear(widths[i], widths[i + 1]))
            layers.append(nn.SiLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        h=self.map(x)
        for res_block in self.res_blocks:
            h = (h + res_block(h)) / np.sqrt(2)
        return h

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module