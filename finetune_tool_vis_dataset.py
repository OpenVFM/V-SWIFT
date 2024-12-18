import warnings
warnings.filterwarnings('ignore')

import argparse
import torch
import numpy as np
import torch
import os
from finetune_dataset_DALI import dali_dataloader
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


def vis_img_and_mask(args, epoch, step, cuda_videos, device, 
                     save_root, mode,
                     chunk_nb=None, split_nb=None, sample_idx=None):

    #NCFHW
    mean_ = torch.as_tensor(args.mean).to(device)[None, :, None, None, None]
    std_ = torch.as_tensor(args.std).to(device)[None, :, None, None, None]
    ori_img = cuda_videos * std_ + mean_

    ori_img_np = ori_img.cpu().numpy()
    ori_img_np = (ori_img_np * 255).astype(np.uint8)
    #NCFHW->NFHWC
    ori_img_np = ori_img_np.transpose((0,2,3,4,1))
    str_temp = "epoch-{}_step-{}_".format(epoch, step)
    if mode == "test":
        test_info = [chunk_nb, split_nb, sample_idx]
    else:
        test_info = None
    save_img(ori_img_np, os.path.join(save_root, mode, str_temp+"ori_img_{}.png"), test_info)


def get_args():
    parser = argparse.ArgumentParser('V-SWIFT fine-tuning and evaluation script for action classification',add_help=False)
    parser.add_argument('--data_root', default='', type=str, help='dataset path root')
    parser.add_argument('--train_data_path', default='', type=str)
    parser.add_argument('--val_data_path', default='', type=str)
    parser.add_argument('--test_data_path', default='', type=str)
    parser.add_argument('--data_set', default='Kinetics-400', type=str)
    parser.add_argument('--nb_classes', default=400, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--sparse_sampling', default=False, action='store_true')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--dali_num_threads', default=2, type=int)
    parser.add_argument('--dali_py_num_workers', default=4, type=int)
    parser.add_argument('--use_decord_bgr', default=False, action='store_true')
    parser.add_argument('--use_random_horizontal_flip', default=False, action='store_true')
    parser.add_argument('--mean', nargs=3, type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument('--std', nargs=3, type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--only_test', action='store_true', help='Perform test evaluation only')
    parser.add_argument('--only_train', action='store_true', help='disable_eval_during_finetuning')
    parser.add_argument('--test_tta_num_segment', type=int, default=2)
    parser.add_argument('--test_tta_num_crop', type=int, default=3)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--print_freq', default=10, type=int, help='step')
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    # Model parameters
    parser.add_argument('--model',default='vit_base_patch16_224',type=str,metavar='MODEL',help='Name of model to train')
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--with_checkpoint', action='store_true', default=False)
    parser.add_argument('--drop',type=float,default=0.0,metavar='PCT',help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate',type=float,default=0.0,metavar='PCT',help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path',type=float,default=0.1,metavar='PCT',help='Drop path rate (default: 0.1)')
    parser.add_argument('--head_drop_rate',type=float,default=0.0,metavar='PCT',help='cls head dropout rate (default: 0.)')
    # Optimizer parameters
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--opt',default='adamw',type=str)
    parser.add_argument('--opt_eps',default=1e-8,type=float)
    parser.add_argument('--opt_betas',default=[0.9, 0.999],type=float,nargs='+',metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad',type=float,default=None,metavar='NORM',help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay',type=float,default=0.05,help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end',type=float,default=None,help="Final value of theweight decay. \
                        We use a cosine schedule for WD and using a larger \
                        decay by the end of training improves performance for ViTs.")
    parser.add_argument('--lr',type=float,default=1e-3,metavar='LR',help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--warmup_lr',type=float,default=1e-8,metavar='LR',help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr',type=float,default=1e-6,metavar='LR',help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs',type=int,default=5,metavar='N',help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps',type=int,default=-1,metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    # Mixup parameters
    parser.add_argument('--reprob',type=float,default=0.25,metavar='PCT',help='Random erase prob (default: 0.25)')
    parser.add_argument('--smoothing',type=float,default=0.1,help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup',type=float,default=0.8,help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix',type=float,default=1.0,help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax',type=float,nargs='+',default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set')
    parser.add_argument('--mixup_prob',type=float,default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob',type=float,default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode',type=str,default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    # dataset torch add
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count')
    parser.add_argument('--resplit', action='store_true', default=False, 
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', 
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
    parser.add_argument('--train_interpolation', type=str, default='bicubic', 
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
 
    mode = "val"
    args.sparse_sampling = True
    save_vis_path = "tool_example/draw_temp/tool_finetune_vis_dataset"
    if not os.path.exists(save_vis_path):
        os.makedirs(save_vis_path)
    os.makedirs(os.path.join(save_vis_path, mode), exist_ok=True)
 
    args.batch_size = 2
    args.dali_num_threads = 2
    args.dali_py_num_workers = 2
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
            files_list.append([None, os.path.join(args.data_root, line_info[0]), int(line_info[1]), -1,-1,-1])

    dali_loader = dali_dataloader(files_list,
                                dali_num_threads = args.dali_num_threads,
                                dali_py_num_workers = args.dali_py_num_workers,
                                batch_size = args.batch_size,
                                input_size = args.input_size,
                                sequence_length = args.num_frames,
                                stride = args.sampling_rate,
                                use_sparse_sampling = args.sparse_sampling,
                                mode = mode,
                                seed = args.seed,
                                args = args)

    with torch.no_grad():
        for epoch in range(0, 30):
            print("epoch: ", epoch)
            for step, dali_batch in enumerate(dali_loader):
                print(step)
                videos = dali_batch[0]
                videos = videos.to(device, non_blocking=True)
                labels = dali_batch[1]
                labels = labels.to(device, non_blocking=True)
                if mode == "test":
                    chunk_nb = dali_batch[2]
                    chunk_nb = chunk_nb.to(device, non_blocking=True)
                    split_nb = dali_batch[3]
                    split_nb = split_nb.to(device, non_blocking=True)
                    sample_idx = dali_batch[4]
                    sample_idx = sample_idx.to(device, non_blocking=True)
                    chunk_nb = chunk_nb.view(-1); split_nb = split_nb.view(-1); sample_idx = sample_idx.view(-1)
                    chunk_nb = chunk_nb.cpu().numpy().tolist()
                    split_nb = split_nb.cpu().numpy().tolist()
                    sample_idx = sample_idx.cpu().numpy().tolist()
                else:
                    chunk_nb, split_nb, sample_idx = None, None, None

                vis_img_and_mask(args, epoch, step, videos, device, 
                                 save_vis_path, mode,
                                 chunk_nb, split_nb, sample_idx)
            if mode == "test":
                exit()
            else:
                dali_loader.reset()