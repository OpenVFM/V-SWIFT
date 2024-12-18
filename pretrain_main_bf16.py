import warnings
warnings.filterwarnings('ignore')

import math
import sys
from typing import Iterable
from torch.nn.utils import clip_grad_norm_
import argparse
import datetime
import json
import os
import time
from pathlib import Path
import torch
from timm.models import create_model
import models
import torch.distributed as dist
from pretrain_dataset_DALI import dali_dataloader
import numpy as np

from tensorboardX import SummaryWriter
import math
import matplotlib.pyplot as plt
from torch import optim as optim
from torch import inf
from pathlib import Path
import random


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


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        decoder_depth=args.decoder_depth,
        with_cp=args.with_checkpoint)
    return model


def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def split_pretrain_to_list(args):
    data_path = args.data_path
    data_root = args.data_root
    
    if not os.path.exists(data_path):
        raise (RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (data_path)))
        
    pretrain_clips = []
    with open(data_path) as split_f:
        data = split_f.readlines()
        for line in data:
            if "," in line:
                line_info = line.strip().split(',')
                assert(len(line_info)==2)
                source_name = "None"
                clip_path = os.path.join(data_root, line_info[0])
                clip_label = int(line_info[1])
                start_frame = -1
                end_frame = -1
                clip_all_frame = -1
            else:
                line_info = line.strip().split(' ')
                if len(line_info) == 4:
                    source_name = "None"
                    clip_path = os.path.join(data_root, line_info[0])
                    clip_label = int(line_info[1])
                    start_frame = int(line_info[2])
                    end_frame = int(line_info[3])
                    clip_all_frame = -1
                elif len(line_info) == 6:
                    source_name = os.path.join(data_root, line_info[0])
                    clip_path = os.path.join(data_root, line_info[1])
                    clip_label = int(line_info[2])
                    start_frame = int(line_info[3])
                    end_frame = int(line_info[4])
                    clip_all_frame = int(line_info[5])
                else:
                    print("format: video_path,video_label")
                    print("format: video_path video_label sframe eframe")
                    print("format: source_name video_path video_label sframe eframe frame_nums")
                    raise (RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
            item = [source_name, clip_path, clip_label, start_frame, end_frame, clip_all_frame]
            pretrain_clips.append(item)
    assert(len(pretrain_clips) != 0)
    return pretrain_clips


class BatchEndCallBackPretrain(object):
    def __init__(
        self,
        world_size,
        batch_size,
        print_freq,
        num_training_steps_per_epoch,
    ):
        self.world_size = world_size
        self.batch_size = batch_size
        self.print_freq = print_freq
        self.num_training_steps_per_epoch = num_training_steps_per_epoch
        
        self.step_time_start = time.time()
        self.init = False
        self.tic = 0
        self.delimiter = "\t"

    def __call__(self, epoch, global_step, min_lr, max_lr, loss_value, grad_norm, weight_decay):
        
        step_idx_cur_epoch = global_step % self.num_training_steps_per_epoch
        
        if (step_idx_cur_epoch % self.print_freq == 0) and (step_idx_cur_epoch != 0):
            if self.init:
                try:
                    speed: float = (
                        self.print_freq * self.batch_size / (time.time() - self.tic)
                    )
                    self.tic = time.time()
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed = float("inf")
                    speed_total = float("inf")

                header = 'Epoch: [{}]'.format(epoch)
                space_fmt = ':' + str(len(str(self.num_training_steps_per_epoch))) + 'd'
                log_msg = [
                    header, 
                    '[{0' + space_fmt + '}/{1}]', 
                    'min_lr: {min_lr}',
                    'lr: {max_lr}',
                    'loss_value: {loss_value}',
                    'grad_norm: {grad_norm}',
                    'weight_decay: {weight_decay}',
                    'video/s/gpu: {qps_v1}', 
                    'video/s: {qps_v2}', 
                    'step_time: {step_time}',
                    'epoch_time: {epoch_time}',
                ]
                log_msg = self.delimiter.join(log_msg)

                step_time = (time.time() - self.step_time_start) / self.print_freq
                epoch_time = step_time * self.num_training_steps_per_epoch
                self.step_time_start = time.time()
                
                min_lr = "{:.10f}".format(min_lr)
                max_lr = "{:.10f}".format(max_lr)
                loss_value = "{:.6f}".format(loss_value)
                grad_norm = "{:.6f}".format(grad_norm) 
                weight_decay = "{:.6f}".format(weight_decay) 
                speed = "{:.6f}".format(speed)
                speed_total = "{:.6f}".format(speed_total)
                step_time = "{:.6f}".format(step_time)
                epoch_time = "{:.6f}".format(epoch_time)
                print(log_msg.format(step_idx_cur_epoch,
                                    self.num_training_steps_per_epoch,
                                    min_lr=str(min_lr),
                                    max_lr=str(max_lr),
                                    loss_value=str(loss_value),
                                    grad_norm=str(grad_norm),
                                    weight_decay=str(weight_decay),
                                    qps_v1=str(speed),
                                    qps_v2=str(speed_total),
                                    step_time=str(step_time),
                                    epoch_time=str(epoch_time)))
            else:
                self.init = True
                self.tic = time.time()
                self.step_time_start = time.time()


class TensorboardLogger(object):

    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v,
                                   self.step if step is None else step)

    def flush(self):
        self.writer.flush()


def save_model_deepspeed(args, epoch, model):
    epoch_name = str(epoch)
    client_state = {'epoch': epoch}
    model.save_checkpoint(
        save_dir=args.output_dir,
        tag="checkpoint-%s" % epoch_name,
        client_state=client_state)


def auto_load_model_deepspeed(args, model):
    output_dir = Path(args.output_dir)
    # deepspeed, only support '--auto_resume'.
    if args.auto_resume:
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
            print("Auto resume checkpoint: %d" % latest_ckpt)
            _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
            if 'epoch' in client_states:
                args.start_epoch = client_states['epoch'] + 1


def save_model_ddp(args,
                epoch,
                model_module,
                optimizer):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_path = os.path.join(output_dir, 'checkpoint-%s.pth' % epoch_name)
    to_save = {
        'model': model_module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    if get_rank() == 0:
        torch.save(to_save, checkpoint_path)


def auto_load_model_ddp(args,
                        model_module,
                        optimizer):
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_module.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("With optim & sched!")


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device)
                         for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), norm_type).to(device)
                for p in parameters
            ]), norm_type)
    return total_norm


def get_parameter_groups(model,
                         weight_decay=1e-5,
                         skip_list=(),
                         get_num_layer=None,
                         get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name.endswith(
                ".scale") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def plot_schedule_values(values, title, xlabel, ylabel, filename):
    """
    Plots the schedule values curve and saves it as a PNG file.

    Parameters:
    values -- a list of schedule values.
    title -- the title of the plot.
    xlabel -- the label for the x-axis.
    ylabel -- the label for the y-axis.
    filename -- the name of the file to save the plot as.
    """
    plt.figure(figsize=(10, 5))  # Set the size of the figure
    plt.plot(values, label='Schedule Values')  # Plot the schedule values curve and add a label
    plt.xlabel(xlabel)  # Set the x-axis label
    plt.ylabel(ylabel)  # Set the y-axis label
    plt.title(title)  # Set the title of the plot
    plt.legend()  # Show the legend
    plt.grid(True)  # Show the grid
    plt.savefig(filename)  # Save the figure as a PNG file
    plt.close()  # Close the figure to avoid displaying it in environments like Jupyter notebook


def load_state_dict(model,
                    state_dict,
                    prefix='',
                    ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".
            format(model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0,
                     warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([
        final_value + 0.5 * (base_value - final_value) *
        (1 + math.cos(math.pi * i / (len(iters)))) for i in iters
    ])

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def main(args):

    # dist init
    args.rank = int(os.environ["RANK"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(
                                    backend = "nccl",
                                    rank = args.rank,
                                    world_size = args.world_size)
    global_rank = get_rank()
    # check local_rank ---> device
    torch.cuda.set_device(args.local_rank)
    device = torch.device(args.local_rank)
    # check print ---> global_rank == 0
    setup_for_distributed(global_rank == 0)

    # check seed in torch numpy random cudnn
    setup_seed(seed=args.seed, cuda_deterministic=False)

    print(args)

    print("use_synthetic: ", args.use_synthetic)
    if not args.use_synthetic:
        files_list = split_pretrain_to_list(args)
    else:
        files_list = None
    print("create train_loader start")
    train_loader = dali_dataloader(files_list,
                                args.dali_num_threads,
                                args.dali_py_num_workers,
                                args.batch_size,
                                gpus_not_equal_num_shards = args.gpus_not_equal_num_shards,
                                input_size = args.input_size,
                                sequence_length = args.num_frames,
                                stride = args.sampling_rate,
                                use_rgb = args.use_decord_bgr,
                                use_flip = args.use_random_horizontal_flip,
                                use_synthetic = args.use_synthetic,
                                set_max_sample = args.set_max_sample,
                                mean = args.mean,
                                std = args.std,
                                seed = args.seed)
    print("create train_loader end")


    print("gpus_not_equal_num_shards: ", args.gpus_not_equal_num_shards)
    if args.gpus_not_equal_num_shards == False:
        args.num_training_steps_per_epoch = int(len(files_list) // args.world_size // args.batch_size)
        args.len_files_list = len(files_list)
        print("num_shards=world_size")
        print("shard_id=rank")
        print("Sampler_train(len(files_list)): ", len(files_list))
    else:
        args.num_training_steps_per_epoch = int(args.set_max_sample // args.world_size // args.batch_size)
        train_loader.step_data_num = args.num_training_steps_per_epoch
        args.len_files_list = args.set_max_sample
        print("num_shards=8")
        print("shard_id=local_rank")
        print("Sampler_train(set_max_sample): ", args.set_max_sample)

    args.total_batch_size = args.world_size * args.batch_size
    args.total_steps = int(args.num_training_steps_per_epoch * args.epochs)
    

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None


    video_model = get_model(args)
    patch_size = video_model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // args.tubelet_size,
                        args.input_size // patch_size[0],
                        args.input_size // patch_size[1])
    args.patch_size = patch_size


    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in ['model', 'module']:
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        load_state_dict(video_model, checkpoint_model)

    video_model.to(device)

    model_module = video_model
    n_parameters = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
    print("model_module = %s" % str(model_module))
    print('number of params:', n_parameters)
 
    weight_decay = args.weight_decay
    if weight_decay:
        skip = {}
        if hasattr(model_module, 'no_weight_decay'):
            skip = model_module.no_weight_decay()
        parameters = get_parameter_groups(model_module, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model_module.parameters()
    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    print("optimizer settings:", opt_args)
    optimizer = optim.AdamW(parameters, **opt_args)

    forward_model = torch.nn.parallel.DistributedDataParallel(
                module=video_model,
                broadcast_buffers=False,
                device_ids=[args.local_rank],
                bucket_cap_mb=32,
                find_unused_parameters=True,
                static_graph=True,
    )
    model_module = forward_model.module


    print("Use step level LR & WD scheduler!")
    lr_schedule_values = cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        args.num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.output_dir and global_rank == 0:
        plot_schedule_values(lr_schedule_values, 
                                            'Learning Rate Schedule', 
                                            'Global step', 
                                            'Learning Rate', 
                                            os.path.join(args.output_dir, 'lr_schedule.png'))
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(args.weight_decay,
                                                args.weight_decay_end,
                                                args.epochs,
                                                args.num_training_steps_per_epoch)
    if args.output_dir and global_rank == 0:
        plot_schedule_values(wd_schedule_values, 
                                            'Weight Decay Schedule', 
                                            'Global step', 
                                            'Weight Decay', 
                                            os.path.join(args.output_dir, 'wd_schedule.png'))
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    auto_load_model_ddp(args=args,
                        model_module=model_module,
                        optimizer=optimizer)
    
    batch_end_callback = BatchEndCallBackPretrain(
        world_size = args.world_size,
        batch_size = args.batch_size,
        print_freq=args.print_freq,
        num_training_steps_per_epoch=args.num_training_steps_per_epoch)

    forward_model.train()
    optimizer.zero_grad()
    global_step = args.start_epoch * args.num_training_steps_per_epoch
    start_time = time.time()

    print("args.start_epoch: ", args.start_epoch)
    print("args.epochs: ", args.epochs)
    print("args.len_files_list: ", args.len_files_list)
    print("batch_size = %d" % (args.batch_size))
    print("args.world_size: ", args.world_size)
    print("num_training_steps_per_epoch = %s" % str(args.num_training_steps_per_epoch))
    print("args.total_steps: ", args.total_steps)
    print("len(lr_schedule_values): ", len(lr_schedule_values))
    print("len(wd_schedule_values): ", len(wd_schedule_values))
    print("global_step: ", global_step)

    while True:
        if global_step % args.num_training_steps_per_epoch == 0:
            if log_writer is not None:
                log_writer.set_step(global_step)

        epoch = global_step // args.num_training_steps_per_epoch
        
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[global_step]

        try:
            dali_batch = next(train_loader)
        except StopIteration:
            train_loader.reset()
            print("rank0 train_loader.reset()")
            dali_batch = next(train_loader)

        videos = dali_batch[0]
        videos = videos.to(device, non_blocking=True)
        labels = dali_batch[1]
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred, target, mask = forward_model(videos, args.mask_ratio)
            B, _, _ = pred.shape
            cal_loss_mask = mask[mask].reshape(B, -1)
            # mae-loss(nn.MSELoss())
            loss = ((((pred - target)**2).mean(dim=-1)) * cal_loss_mask).sum() / cal_loss_mask.sum()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(2)

        loss.backward()
        if args.clip_grad is not None:
            grad_norm = clip_grad_norm_(forward_model.parameters(), max_norm=args.clip_grad)
        else:
            grad_norm = get_grad_norm_(forward_model.parameters())
        optimizer.step()
        optimizer.zero_grad()

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]

        if log_writer is not None:
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.set_step()

        batch_end_callback(epoch, global_step,  min_lr,  max_lr, loss_value, grad_norm, weight_decay_value)

        if global_step % args.num_training_steps_per_epoch == 0 and epoch > 0:
            if args.output_dir:
                _epoch = epoch + 1
                if _epoch % args.save_ckpt_freq == 0 or _epoch == args.epochs:
                    save_model_ddp(args=args, epoch=epoch,
                                model_module=model_module, optimizer=optimizer)

            if args.output_dir and global_rank == 0:
                if log_writer is not None:
                    log_writer.flush()

                log_stats = {
                    'epoch': epoch,
                    "train_min_lr": "{:.10f}".format(min_lr),
                    "train_lr": "{:.10f}".format(max_lr),   
                    "train_loss": "{:.6f}".format(loss_value), 
                    'grad_norm': "{:.6f}".format(grad_norm), 
                    'weight_decay': "{:.6f}".format(weight_decay_value), 
                    'n_parameters': n_parameters
                }
                with open(os.path.join(args.output_dir, "log.txt"),mode="a",encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
                    
        global_step += 1

        if global_step == args.total_steps:
            save_model_ddp(args=args, epoch=epoch,
                        model_module=model_module, optimizer=optimizer)
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            exit()
            

if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)