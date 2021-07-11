# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torchvision.transforms as T

from .transforms import *
from .autoaugment import AutoAugment


def build_transforms(cfg, is_train=True):
    res = []

    if is_train:
        size_train = cfg.INPUT.SIZE_TRAIN

        # crop
        do_crop = cfg.INPUT.CROP.ENABLED
        crop_size = cfg.INPUT.CROP.SIZE
        crop_scale = cfg.INPUT.CROP.SCALE
        crop_ratio = cfg.INPUT.CROP.RATIO

        # augmix augmentation
        do_augmix = cfg.INPUT.AUGMIX.ENABLED
        augmix_prob = cfg.INPUT.AUGMIX.PROB

        # auto augmentation
        do_autoaug = cfg.INPUT.AUTOAUG.ENABLED
        autoaug_prob = cfg.INPUT.AUTOAUG.PROB

        # horizontal filp
        do_flip = cfg.INPUT.FLIP.ENABLED
        flip_prob = cfg.INPUT.FLIP.PROB

        # padding
        do_pad = cfg.INPUT.PADDING.ENABLED
        padding_size = cfg.INPUT.PADDING.SIZE
        padding_mode = cfg.INPUT.PADDING.MODE

        # color jitter
        do_cj = cfg.INPUT.CJ.ENABLED
        cj_prob = cfg.INPUT.CJ.PROB
        cj_brightness = cfg.INPUT.CJ.BRIGHTNESS
        cj_contrast = cfg.INPUT.CJ.CONTRAST
        cj_saturation = cfg.INPUT.CJ.SATURATION
        cj_hue = cfg.INPUT.CJ.HUE
        print(cj_prob, cj_saturation)
        # random affine
        do_affine = cfg.INPUT.AFFINE.ENABLED

        # random erasing
        do_rea = cfg.INPUT.REA.ENABLED
        rea_prob = cfg.INPUT.REA.PROB
        rea_value = cfg.INPUT.REA.VALUE

        # random patch
        do_rpt = cfg.INPUT.RPT.ENABLED
        rpt_prob = cfg.INPUT.RPT.PROB

        do_gauss_noise = True  # cfg.INPUT.NOISE.ENABLED
        gauss_noise_var = 4.8  # cfg.INPUT.NOISE.VAR

        if do_autoaug:
            res.append(T.RandomApply([AutoAugment()], p=autoaug_prob))

        if size_train[0] > 0:
            res.append(T.Resize(size_train[0] if len(size_train) == 1 else size_train, interpolation=3))

        if do_affine:
            res.append(T.RandomAffine(degrees=15, shear=15, resample=3))

        if do_crop:
            res.append(T.RandomResizedCrop(size=crop_size[0] if len(crop_size) == 1 else crop_size,
                                           interpolation=3,
                                           scale=crop_scale, ratio=crop_ratio))
        if do_pad:
            res.extend([T.Pad(padding_size, padding_mode=padding_mode),
                        T.RandomCrop(size_train[0] if len(size_train) == 1 else size_train)])
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))

        if do_cj:
            res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
        if do_affine:
            res.append(T.RandomAffine(degrees=10, translate=None, scale=[0.9, 1.1], shear=0.1, resample=False,
                                      fillcolor=0))
        if do_augmix:
            res.append(AugMix(prob=augmix_prob))

        if True:
            def get_compress(size_in, size_out):
                return T.Compose([
                    T.Resize((size_in, size_in), interpolation=3),
                    T.Resize((size_out, size_out), interpolation=3),
                ])

            probs = torch.tensor([0.2, 0.8])
            sizes = [get_compress( 84, 256),
                     get_compress(128, 256),
                     get_compress(64, 256)]

            def compress(image, p=0.1):
                if torch.rand(1) < p:
                    return sizes[(probs < torch.rand(1)).sum()](image)
                return image

            res.append(compress)

        res.append(ToTensor())
        if do_gauss_noise:
            def gauss_noise(tensor):
                return torch.clip(tensor + torch.normal(0, gauss_noise_var, size=tensor.shape), 0, 255)
            res.append(T.Lambda(gauss_noise))
        if do_rea:
            res.append(T.RandomErasing(p=rea_prob, value=rea_value))
        if do_rpt:
            res.append(RandomPatch(prob_happen=rpt_prob))
    else:
        size_test = cfg.INPUT.SIZE_TEST
        do_crop = cfg.INPUT.CROP.ENABLED
        crop_size = cfg.INPUT.CROP.SIZE

        if size_test[0] > 0:
            res.append(T.Resize(size_test[0] if len(size_test) == 1 else size_test, interpolation=3))
        if do_crop:
            res.append(T.CenterCrop(size=crop_size[0] if len(crop_size) == 1 else crop_size))
        res.append(ToTensor())
    return T.Compose(res)
