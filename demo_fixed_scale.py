import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
import models

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

def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell)
            ql = qr
            preds.append(pred)
        pred = torch.cat(preds, dim=2)
    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    # parser.add_argument('--model', default='pretrained_model/mambasr-edsr.pth')
    parser.add_argument('--model', default='pretrained_model/mambasr-rdn.pth')
    parser.add_argument('--scale', default='2')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    scale_max = 4
    
    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    h = int(img.shape[-2] * int(args.scale))
    w = int(img.shape[-1] * int(args.scale))
    scale = h / img.shape[-2]
    coord = make_coord((h, w), flatten=False).cuda()
    cell = torch.ones(1,2).cuda()
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    
    cell_factor = max(scale/scale_max, 1)
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
       coord.unsqueeze(0), cell_factor*cell, bsize=300).squeeze(0)
    # pred = model(((img - 0.5) / 0.5).cuda().unsqueeze(0),coord.unsqueeze(0), cell_factor*cell).squeeze(0)
    pred = (pred * 0.5 + 0.5).clamp(0, 1).reshape(3, h, w).cpu()
    transforms.ToPILImage()(pred).save(args.output)
