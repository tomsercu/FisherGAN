Fisher GAN
==========

[PyTorch](http://pytorch.org) code accompanying the paper ["Fisher GAN"](https://arxiv.org/abs/1705.09675)

Tested on `version 0.1.12`.

To reproduce LSUN results (identical for CelebA, but set higher `--niter`):

```
python main.py --dataset lsun --dataroot <lsun-root> --cuda --Diters 2 --adam --lrG 2e-4 --lrD 2e-4 --G_extra_layers 2
```

For CIFAR-10:

```
python main.py --dataset cifar10 --dataroot <cifar10-root> --niter 350 --cuda --Diters 2 --adam --lrG 2e-4 --lrD 2e-4 --imageSize 32 --G_extra_layers 2 --D_extra_layers 2 
```
