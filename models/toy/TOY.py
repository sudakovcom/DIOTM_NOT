import torch
import torch.nn as nn
import numpy as np
import math

# class ToyDiscriminator(nn.Module):
#     def __init__(self, data_dim=2, hidden_dim=256, num_timesteps=4):
#         super().__init__()
#         self.out_module = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, 1),
#         )
#         self.num_timesteps = num_timesteps

#         self.x_module = nn.Sequential(
#             nn.Linear(data_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )

#         self.t_module = nn.Sequential(
#             nn.Linear(128, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )
    
#     def forward(self, x, t):
#         t = self.num_timesteps * t
#         t = get_timestep_embedding(t)
#         temb = self.t_module(t)
#         xemb = self.x_module(x)
#         return self.out_module(xemb + temb)

class ToyDiscriminator(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128, num_timesteps=4):
        super().__init__()
        hid = hidden_dim
        self.x_module = nn.Sequential(
            nn.Linear(data_dim+1, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
            nn.SiLU(),
            nn.Linear(hid, 1),
        )
        self.num_timesteps = num_timesteps
    
    def forward(self, x, t):
        x = x.float()
        x_out = torch.cat([x, t[:,None]], dim=1)
        return self.x_module(x_out)


class ToyGenerator(torch.nn.Module):
    def __init__(self, data_dim=2, hidden_dim=256, z_dim=2, num_timesteps=4):
        super(ToyGenerator, self).__init__()
        hid = hidden_dim

        self.xz_module = nn.Sequential(
            nn.Linear(data_dim+z_dim, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
            nn.SiLU(),
            nn.Linear(hid, data_dim),
        )

        self.scale_module = nn.Sequential(
            nn.Linear(data_dim+z_dim, hid),
            nn.SiLU(),
            nn.Linear(hid, data_dim),
        )

        self.num_timesteps = num_timesteps

    def forward(self, x, z):
        # scale = self.scale_module(torch.cat([x,z], dim=1)) + 1
        return x + self.xz_module(torch.cat([x,z], dim=1))
        


# class ToyGenerator(torch.nn.Module):
#     def __init__(self, data_dim=2, hidden_dim=128, z_dim=2, num_timesteps=4):
#         super(ToyGenerator, self).__init__()
#         hid = hidden_dim

#         self.z_module = nn.Sequential(
#             nn.Linear(z_dim, hid),
#             nn.SiLU(),
#             nn.Linear(hid, hid),
#         )

#         self.x_module = ResNet_FC(data_dim, hidden_dim, num_res_blocks=3)

#         self.out_module = nn.Sequential(
#             nn.Linear(hid,hid),
#             nn.SiLU(),
#             nn.Linear(hid, data_dim),
#         )

#         self.num_timesteps = num_timesteps

#     def forward(self, x, z):
#         z_out = self.z_module(z)
#         x_out = self.x_module(x)
#         out   = self.out_module(x_out+z_out)
        # return out


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