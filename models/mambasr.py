import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import models
from models.galerkin import simple_attn
from models import register
from models.mamba_basic import ResidualGroup

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

@register('mambasr')
class MAMBASR(nn.Module):
    def __init__(self, encoder_spec, width=256, blocks=16):
        super().__init__()
        self.width = width
        self.encoder = models.make(encoder_spec)
        self.conv00 = nn.Conv2d((64 + 2)*4+2, self.width, 1)

        self.conv0 = simple_attn(self.width, blocks)
        self.conv1 = simple_attn(self.width, blocks)
        self.num_layers = 2
        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()        
        for i_layer in range(self.num_layers):
            layer = ResidualGroup(dim=64, input_resolution=48, depth=6, mlp_ratio=2, img_size=48, patch_size=8)
            self.layers.append(layer)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Conv2d(self.width, 256, 1)
        self.fc2 = nn.Conv2d(256, 3, 1)
        
    def gen_feat(self, inp):
        self.inp = inp
        x = self.encoder(inp)
        x_size = (x.shape[2], x.shape[3])
        for layer in self.layers:
            x = layer(x, x_size)
        self.feat = x.transpose(1, 2).view(x.shape[0], -1, x_size[0], x_size[1])
        return self.feat
        
    def query_rgb(self, coord, cell):      
        feat = (self.feat)
        grid = 0

        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:

                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)

                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                rel_coords.append(rel_coord)
                feat_s.append(feat_)
                
        rel_cell = cell.clone()
        rel_cell[:,0] *= feat.shape[-2]
        rel_cell[:,1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)
         
        grid = torch.cat([*rel_coords, *feat_s, \
            rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1,1,coord.shape[1],coord.shape[2])],dim=1)

        x = self.conv00(grid)
        x = self.conv0(x, 0)
        x = self.conv1(x, 1)
        feat = x
        ret = self.fc2(F.gelu(self.fc1(feat)))
        ret = ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
                                padding_mode='border', align_corners=False)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

