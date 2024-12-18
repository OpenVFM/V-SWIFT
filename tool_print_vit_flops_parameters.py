import timm
import torch
import numpy as np
import random
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from timm.models import create_model
import models

finetune_model_list = [
    'pixel_pretrain_videomae_small_patch16_224',
    'pixel_pretrain_videomae_base_patch16_224',
    'pixel_pretrain_videomae_large_patch16_224',
    'pixel_pretrain_videomae_huge_patch16_224',
    'pixel_pretrain_videomae_giant_patch14_224',
]

seed = 4217
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_frames = [16]
tubelet_size = [2]
for _tubelet_size in tubelet_size:
    for _num_frames in num_frames:
        for model_name in finetune_model_list:
            backbone = timm.create_model(model_name,
                                        all_frames=_num_frames,
                                        tubelet_size=_tubelet_size,
                                        pretrained=False)

            img_size = model_name.split('_')[-1]
            img_size = int(img_size)

            backbone.cuda()
            videos = torch.rand(1, 3, _num_frames, img_size, img_size).cuda()
            flops = FlopCountAnalysis(backbone, (videos, 0.9))

            print("_tubelet_size:", _tubelet_size)
            print("_num_frames:", _num_frames)
            print("model_name:", model_name)
            print(flop_count_table(flops, max_depth=1))