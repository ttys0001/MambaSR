# MambaSR
This repository contains the official implementation for MambaSR

Our code is based on Ubuntu 18.04, pytorch 1.13.1, CUDA 11.7, causal-conv1d 1.0.0, mamba_ssm 1.0.1 and python 3.9.

## Demo
### Fixed Scale
`python demo_fixed_scale.py --input input.png --model pretrained_model/mambasr-edsr.pth --scale 2 --output output_fixed_scale.png`

### Arbitrary Scale
`python demo_arbitrary_scale.py --input input.png --model pretrained_model/mambasr-edsr.pth --height 1000 --weight 1280 --output output_arbitrary_scale.png`
