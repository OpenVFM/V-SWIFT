import warnings
warnings.filterwarnings('ignore')

import argparse
import torch
import numpy as np
import torch
import einops
import os
import torch.distributed as dist
from pretrain_dataset_DALI import dali_dataloader
from timm.models import create_model
import models
from PIL import Image

def save_img(images_np, save_path, test_info=None):
    resized_images_list = []
    for b in range(images_np.shape[0]):  # batch
        resized_batch = []
        for t in range(images_np.shape[1]):  # time
            image = Image.fromarray(images_np[b, t].astype('uint8'))
            image_resized = image.resize((128, 128))
            resized_batch.append(np.array(image_resized))
        resized_images_list.append(resized_batch)

    merged_images = []
    for batch_images in resized_images_list:
        merged_image = Image.new('RGB', (4 * 128, 4 * 128))
        for idx, img_array in enumerate(batch_images):
            x = idx % 4 * 128
            y = idx // 4 * 128
            merged_image.paste(Image.fromarray(img_array), (x, y))
        merged_images.append(merged_image)

    if test_info is not None:
        chunk_nb, split_nb, sample_idx = test_info
        for i, merged_image in enumerate(merged_images):
            timeidx = chunk_nb[i]
            spaceidx = split_nb[i]
            videoidx = sample_idx[i]
            directory_path = os.path.dirname(save_path.format(i))
            filename = os.path.basename(save_path.format(i))
            os.makedirs(os.path.join(directory_path, str(videoidx)), exist_ok=True)
            test_save_path = os.path.join(directory_path, str(videoidx), filename)
            merged_image.save(test_save_path)
    else:
        for i, merged_image in enumerate(merged_images):
            merged_image.save(save_path.format(i))


def vis_img_and_mask(args, epoch, step, cuda_videos, cuda_bool_masked_pos, device, save_root):

    #NCFHW
    mean_ = torch.as_tensor(args.mean).to(device)[None, :, None, None, None]
    std_ = torch.as_tensor(args.std).to(device)[None, :, None, None, None]
    ori_img = cuda_videos * std_ + mean_


    ori_img_np = ori_img.cpu().numpy()
    ori_img_np = (ori_img_np * 255).astype(np.uint8)
    #NCFHW->NFHWC
    ori_img_np = ori_img_np.transpose((0,2,3,4,1))
    str_temp = "epoch-{}_step-{}_".format(epoch, step)
    save_img(ori_img_np, os.path.join(save_root, "ori", str_temp+"ori_img_{}.png"))


    #====================================================================
    tubelet_size = args.tubelet_size
    window_size = args.window_size
    patch_size = args.patch_size
    #NCFHW
    img_patch = einops.rearrange(ori_img, 
                                'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', 
                                p0=tubelet_size, p1=patch_size[0], p2=patch_size[0])
    img_patch = einops.rearrange(img_patch, 
                                'b n p c -> b n (p c)')
    mask = torch.ones_like(img_patch)
    mask[cuda_bool_masked_pos] = 0
    mask = einops.rearrange(mask, 
                            'b n (p c) -> b n p c', c=3)
    mask = einops.rearrange(mask, 
                            'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ', 
                            p0=tubelet_size, p1=patch_size[0], p2=patch_size[1], h=window_size[-2], w=window_size[-1])
    img_mask = ori_img * mask
    img_mask_np = img_mask.cpu().numpy()
    img_mask_np = (img_mask_np * 255).astype(np.uint8)
    #NCFHW->NFHWC
    img_mask_np = img_mask_np.transpose((0,2,3,4,1))
    str_temp = "epoch-{}_step-{}_".format(epoch, step)
    save_img(img_mask_np, os.path.join(save_root, "mask", str_temp+"mask_img_{}.png"))

def get_args():
    parser = argparse.ArgumentParser('V-SWIFT pre-training script', add_help=False)
    # base parameters
    parser.add_argument('--data_root', default='', type=str, help='dataset path root')
    parser.add_argument('--data_path', default='', type=str, help='dataset txt or csv')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--dali_num_threads', default=2, type=int)
    parser.add_argument('--dali_py_num_workers', default=4, type=int)
    parser.add_argument('--use_decord_bgr', default=False, action='store_true')
    parser.add_argument('--use_random_horizontal_flip', default=False, action='store_true')
    parser.add_argument('--mean', nargs=3, type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument('--std', nargs=3, type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--print_freq', default=10, type=int, help='step')
    parser.add_argument('--save_ckpt_freq', default=20, type=int, help='epoch')
    # test MFU(synthetic data)
    parser.add_argument('--use_synthetic', default=False, action='store_true')
    # Solution for Limited Storage Space
    parser.add_argument('--gpus_not_equal_num_shards', default=False, action='store_true')
    parser.add_argument('--set_max_sample', default=160000, type=int)
    # Model parameters
    parser.add_argument('--model', default='pixel_pretrain_videomae_base_patch16_224', type=str, metavar='MODEL')
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--with_checkpoint', action='store_true', default=False)
    parser.add_argument('--decoder_depth', default=4, type=int, help='depth of decoder')
    parser.add_argument('--mask_ratio', default=0.9, type=float, help='mask ratio of encoder')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT', help='Drop path rate')
    # Optimizer parameters
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon')
    parser.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+', metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--clip_grad',type=float,default=None,metavar='NORM',help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay',type=float,default=0.05,help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end',type=float,default=None,help="Final value of theweight decay. \
                        We use a cosine schedule for WD and using a larger \
                        decay by the end of training improves performance for ViTs.")
    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR', help='learning rate')
    parser.add_argument('--warmup_lr',type=float,default=1e-6,metavar='LR',help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs',type=int,default=40,metavar='N',help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps',type=int,default=-1,metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
 
    save_vis_path = "tool_example/draw_temp/tool_pretrain_vis_dataset"
    if not os.path.exists(save_vis_path):
        os.makedirs(save_vis_path) 
    os.makedirs(os.path.join(save_vis_path, "mask"), exist_ok=True)
    os.makedirs(os.path.join(save_vis_path, "ori"), exist_ok=True)

    args.batch_size = 2
    args.dali_num_threads = 2
    args.dali_py_num_workers = 2
    args.mask_ratio = 0.6

    args.rank = 0
    args.local_rank = 0
    args.world_size = 1
    torch.distributed.init_process_group(
        backend = "nccl",
        init_method = "tcp://127.0.0.1:12584",
        rank = args.rank,
        world_size = args.world_size)
    torch.cuda.set_device(args.local_rank)
    device = torch.device(args.local_rank)
     
    args.data_path = "tool_example/k400_sample.csv"
    args.data_root = "tool_example/k400_sample"
    files_list = []
    with open(args.data_path) as split_f:
        data = split_f.readlines()
        for line in data:
            line_info = line.strip().split(',')
            files_list.append((None, os.path.join(args.data_root, line_info[0]), int(line_info[1]), -1,-1,-1))
            
    train_loader = dali_dataloader(files_list,
                                args.dali_num_threads,
                                args.dali_py_num_workers,
                                args.batch_size,
                                input_size = args.input_size,
                                sequence_length = args.num_frames,
                                stride = args.sampling_rate,
                                use_rgb = args.use_decord_bgr,
                                use_flip = args.use_random_horizontal_flip,
                                mean = args.mean,
                                std = args.std)

    model = create_model(
            args.model,
            pretrained=False,
            all_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            decoder_depth=args.decoder_depth,
            with_cp=args.with_checkpoint).cuda()
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // args.tubelet_size,
                        args.input_size // patch_size[0],
                        args.input_size // patch_size[1])
    args.patch_size = patch_size


    with torch.no_grad():
        for epoch in range(0, 30):
            print("epoch: ", epoch)
            for step, dali_batch in enumerate(train_loader):
                print(step)
                
                videos = dali_batch[0]
                videos = videos.to(device, non_blocking=True)

                labels = dali_batch[1]
                labels = labels.to(device, non_blocking=True)

                pred, target, mask = model(videos, args.mask_ratio)

                vis_img_and_mask(args, epoch, step, videos, mask, device, save_vis_path)

            train_loader.reset()