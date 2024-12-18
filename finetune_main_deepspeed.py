import warnings
warnings.filterwarnings('ignore')

import argparse
import datetime
import json
import os
import sys
import glob
import random
import time
from collections import OrderedDict
from collections import defaultdict
from pathlib import Path
import deepspeed
import numpy as np
import torch
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torch.distributed as dist

from timm.models import create_model
import models

if os.getenv('DALI') == '1':
    from finetune_dataset_DALI import dali_dataloader
elif os.getenv('TORCH') == '1':
    from finetune_dataset_torch import dali_dataloader
else:
    print("error: finetune_dataset aug method: {}=1  or {}=1".format('DALI', 'TORCH'))
    sys.exit(2)

from tensorboardX import SummaryWriter
import math
import matplotlib.pyplot as plt
from timm.utils import accuracy
from scipy.special import softmax
from multiprocessing import Pool
from collections import defaultdict, deque
from torch.nn.utils import clip_grad_norm_
from torch import optim as optim
from torch import inf

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

    # enable_deepspeed
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    known_args, _ = parser.parse_known_args()
    if known_args.enable_deepspeed:
        parser = deepspeed.add_config_arguments(parser)
        ds_init = deepspeed.initialize
    else:
        ds_init = None
    return parser.parse_args(), ds_init


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        img_size=args.input_size,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        head_drop_rate=args.head_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        with_cp=args.with_checkpoint,
    )
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


def split_finetune_to_list(args):
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    test_data_path = args.test_data_path
    data_root = args.data_root

    if not os.path.exists(train_data_path):
        raise (RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (train_data_path)))
    if not os.path.exists(val_data_path):
        raise (RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (val_data_path)))
    if not os.path.exists(test_data_path):
        raise (RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (test_data_path)))
    
    data_clips = defaultdict(list)
    for data_name, data_path in zip(["train", "val", "test"], 
                         [train_data_path, val_data_path, test_data_path]):
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
                data_clips[data_name].append(item)
        assert(len(data_clips[data_name]) != 0)
    return data_clips["train"], data_clips["val"], data_clips["test"]


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64,
                         device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def min(self):
        return min(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            min=self.min,
            value=self.value)


class MetricLogger(object):

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

        self.step_time_start = 0
        self.init = False
        self.tic = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, 
                        world_size=None, batch_size=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f} ({min:.4f} -- {max:.4f})')
        data_time = SmoothedValue(fmt='{avg:.4f} ({min:.4f} -- {max:.4f})')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header, '[{0' + space_fmt + '}/{1}]', 
            'eta: {eta}', 
            '{meters}',
            'time: {time}', 
            'data: {data}',
            'max mem: {memory:.0f}']

        if (world_size is not None) and (batch_size is not None):
            log_msg.append('video/s/gpu: {qps_v1}')
            log_msg.append('video/s: {qps_v2}')

        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                if self.init:
                    if (world_size is not None) and (batch_size is not None):
                        try:
                            speed = print_freq * batch_size / (time.time() - self.tic)
                            self.tic = time.time()
                            speed_total = speed * world_size
                        except ZeroDivisionError:
                            speed = float("inf")
                            speed_total = float("inf")

                    eta_seconds = iter_time.global_avg * (len(iterable) - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                    if (world_size is not None) and (batch_size is not None):
                        speed = "{:.4f}".format(speed)
                        speed_total = "{:.4f}".format(speed_total)

                        print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self),
                            time=str(iter_time), data=str(data_time), memory=torch.cuda.max_memory_allocated() / MB,
                            qps_v1=str(speed), qps_v2=str(speed_total)))
                    else:
                        print(
                            log_msg.format(i, len(iterable), eta=eta_string, meters=str(self),
                                time=str(iter_time), data=str(data_time), memory=torch.cuda.max_memory_allocated() / MB))

                else:
                    self.init = True
                    self.tic = time.time()
                    self.step_time_start = time.time()
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / len(iterable)))


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


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):

    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


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


def load_finetune_checkpoint(args, video_model):
    checkpoint = torch.load(args.finetune, map_location='cpu')
    print("Load ckpt from %s" % args.finetune)
    
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
        
    for old_key in list(checkpoint_model.keys()):
        if old_key.startswith('_orig_mod.'):
            new_key = old_key[10:]
            checkpoint_model[new_key] = checkpoint_model.pop(old_key)

    state_dict = video_model.state_dict()
    
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            if checkpoint_model[k].shape[0] == 710 and args.data_set.startswith('Kinetics'):
                print(f'Convert K710 head to {args.data_set} head')
                if args.data_set == 'Kinetics-400':
                    label_map_path = 'misc/label_710to400.json'
                elif args.data_set == 'Kinetics-600':
                    label_map_path = 'misc/label_710to600.json'
                elif args.data_set == 'Kinetics-700':
                    label_map_path = 'misc/label_710to700.json'
                label_map = json.load(open(label_map_path))
                checkpoint_model[k] = checkpoint_model[k][label_map]
            else:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict
    
    if 'pos_embed' in checkpoint_model:
        print("if 'pos_embed' in checkpoint_model")

    load_state_dict(video_model, checkpoint_model, prefix=args.model_prefix)
    
    return video_model


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


def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir,
                                         "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        args.opt_betas[0],
                        args.opt_betas[1]
                    ],
                    "eps": args.opt_eps
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        if args.clip_grad is not None:
            ds_config.update({'gradient_clipping': args.clip_grad})

        writer.write(json.dumps(ds_config, indent=2))


def main(args, ds_init):

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

    if ds_init is not None:
        create_ds_config(args)

    # check seed in torch numpy random cudnn
    setup_seed(seed=args.seed, cuda_deterministic=False)

    print(args)

    train_files_list, val_files_list, test_files_list = split_finetune_to_list(args)
    nb_classes_map = {
        'Kinetics-400': 400,
        'Kinetics-600': 600,
        'Kinetics-700': 700,
        'Kinetics-710': 710,
        'SSV2': 174,
        'UCF101': 101,
        'HMDB51': 51,
        'Diving48': 48,
        'MIT': 339
    }
    assert(args.nb_classes == nb_classes_map[args.data_set])
    print("create train_loader start")
    train_loader = dali_dataloader(train_files_list,
                                    args.dali_num_threads,
                                    args.dali_py_num_workers,
                                    args.batch_size,
                                    input_size = args.input_size,
                                    sequence_length = args.num_frames,
                                    stride = args.sampling_rate,
                                    use_sparse_sampling = args.sparse_sampling,
                                    mode = "train",
                                    seed = args.seed,
                                    args = args)
    print("create train_loader end")
    args.len_files_list = len(train_files_list)
    args.num_training_steps_per_epoch = int(len(train_files_list) // args.world_size // args.batch_size)
    args.train_total_batch_size = args.world_size * args.batch_size

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes)

    video_model = get_model(args)

    patch_size = video_model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // args.tubelet_size,
                        args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size


    video_model = load_finetune_checkpoint(args, video_model)

    video_model.to(device)

    model_module = video_model
    n_parameters = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
    print("model_module = %s" % str(model_module))
    print('number of params:', n_parameters)

    num_layers = model_module.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay**(num_layers + 1 - i)for i in range(num_layers + 2)))
    else:
        assigner = None
    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model_module.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)
   
    if ds_init is not None:
        optimizer_params = get_parameter_groups(
                video_model, args.weight_decay, skip_weight_decay_list,
                assigner.get_layer_id if assigner is not None else None,
                assigner.get_scale if assigner is not None else None)
        forward_model, optimizer, _, _ = ds_init(args=args, model=video_model,
                                model_parameters=optimizer_params, dist_init_required=False)
    else:
        forward_model = torch.nn.parallel.DistributedDataParallel(
            module=video_model,
            broadcast_buffers=False,
            device_ids=[args.local_rank],
            bucket_cap_mb=32,
            find_unused_parameters=True,
            static_graph=True,
        )
        model_module = forward_model.module

        # create_optimizer
        get_num_layer = assigner.get_layer_id if assigner is not None else None
        get_layer_scale = assigner.get_scale if assigner is not None else None
        weight_decay = args.weight_decay
        if weight_decay:
            skip = {}
            if skip_weight_decay_list is not None:
                skip = skip_weight_decay_list
            elif hasattr(model_module, 'no_weight_decay'):
                skip = model_module.no_weight_decay()
            parameters = get_parameter_groups(model_module, weight_decay, skip,
                                            get_num_layer, get_layer_scale)
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


    print("Use step level LR scheduler!")
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

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))

    if ds_init is not None:
        auto_load_model_deepspeed(args=args, model=forward_model)
    else:
        auto_load_model_ddp(args=args,
                            model_module=model_module,
                            optimizer=optimizer)
    
    
    if args.only_test:
        print("create test_loader start")
        test_loader = dali_dataloader(test_files_list,
                                    args.dali_num_threads,
                                    args.dali_py_num_workers // 2,
                                    args.batch_size // 2,
                                    input_size = args.input_size,
                                    sequence_length = args.num_frames,
                                    stride = args.sampling_rate,
                                    use_sparse_sampling = args.sparse_sampling,
                                    mode = "test",
                                    seed = args.seed,
                                    args = args)
        tta_nums = args.test_tta_num_segment * args.test_tta_num_crop
        args.test_steps_per_epoch = int((len(test_files_list) * tta_nums)  // args.world_size // (args.batch_size // 2))
        args.test_total_batch_size = args.world_size * (args.batch_size // 2)
        test_loader.step_data_num = args.test_steps_per_epoch
        args.test_files_list = len(test_files_list)
        print("create test_loader end")
        print("test videos: ", args.test_files_list)
        print("test tta videos: ", (len(test_files_list) * tta_nums))
        
        preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
        test_stats = final_test(test_loader, forward_model, device, preds_file, args)
        torch.distributed.barrier()
        if global_rank == 0:
            print("Start merging results...")
            final_top1, final_top5 = merge(args.output_dir, num_tasks = args.world_size)
            print(f"Accuracy of the network on the {args.test_files_list} test videos: \
                Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            log_stats = {'Final top-1': final_top1, 'Final Top-5': final_top5}
            if args.output_dir:
                with open(os.path.join(args.output_dir, "log.txt"), mode = "a", encoding = "utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        exit(0)     

    if not args.only_train:
        print("create val_loader start")
        val_loader = dali_dataloader(val_files_list,
                                    args.dali_num_threads,
                                    args.dali_py_num_workers // 2,
                                    args.batch_size // 2,
                                    input_size = args.input_size,
                                    sequence_length = args.num_frames,
                                    stride = args.sampling_rate,
                                    use_sparse_sampling = args.sparse_sampling,
                                    mode = "val",
                                    seed = args.seed,
                                    args = args)
        args.val_steps_per_epoch = int(len(val_files_list) // args.world_size // (args.batch_size // 2))
        args.val_total_batch_size = args.world_size * (args.batch_size // 2)
        args.val_files_list = len(val_files_list)
        print("create val_loader end")
    else:
        val_loader = None


    print("args.start_epoch: ", args.start_epoch)
    print("args.epochs: ", args.epochs)
    print("args.len_files_list: ", args.len_files_list)
    print("batch_size = %d" % (args.batch_size))
    print("args.world_size: ", args.world_size)
    print("num_training_steps_per_epoch = %s" % str(args.num_training_steps_per_epoch))
    print("len(lr_schedule_values): ", len(lr_schedule_values))
    print("len(wd_schedule_values): ", len(wd_schedule_values))
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    print('number of params: {}'.format(n_parameters))


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * args.num_training_steps_per_epoch)

        forward_model.train(True)
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        if ds_init is not None:
            forward_model.zero_grad()
            forward_model.micro_steps = 0
        else:
            optimizer.zero_grad()

        start_steps = int(epoch * args.num_training_steps_per_epoch)
        for data_iter_step, (videos, labels) in enumerate(metric_logger.log_every(
                                                        train_loader, args.print_freq, 
                                                        header, 
                                                        args.world_size, args.batch_size)):
            it = start_steps + data_iter_step
            # Update LR & WD for the first acc
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels = labels.view(-1)

            if mixup_fn is not None:
                # mixup handle 3th & 4th dimension
                B, C, T, H, W = videos.shape
                videos = videos.view(B, C * T, H, W)
                videos, labels = mixup_fn(videos, labels)
                videos = videos.view(B, C, T, H, W)
            

            if ds_init is not None:
                videos = videos.half()
                outputs = forward_model(videos)
                loss = criterion(outputs, labels)
            else:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = forward_model(videos)
                    loss = criterion(outputs, labels)


            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)


            if ds_init is not None:
                forward_model.backward(loss)
                grad_norm = forward_model.get_global_grad_norm()
                forward_model.step()
                loss_scale_value = optimizer.loss_scale if hasattr(optimizer, "loss_scale") \
                                                                else optimizer.cur_scale
            else:
                optimizer.zero_grad()
                loss.backward()
                if args.clip_grad is not None:
                    grad_norm = clip_grad_norm_(forward_model.parameters(), max_norm=args.clip_grad)
                else:
                    grad_norm = get_grad_norm_(forward_model.parameters())
                optimizer.step()
                loss_scale_value = 0.0
            
            
            if mixup_fn is None:
                class_acc = (outputs.max(-1)[-1] == labels).float().mean()
            else:
                class_acc = None
            metric_logger.update(loss=loss_value)
            metric_logger.update(class_acc=class_acc)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(class_acc=class_acc, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.set_step()
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        # update DALI loader
        train_loader.reset()

        if args.output_dir:
            _epoch = epoch + 1
            if _epoch % args.save_ckpt_freq == 0 or _epoch == args.epochs:
                if ds_init is not None:
                    save_model_deepspeed(args=args, model=forward_model, epoch=epoch)
                else:
                    save_model_ddp(args=args,
                                    epoch=epoch,
                                    model_module=model_module,
                                    optimizer=optimizer)

        if val_loader is not None:
            test_stats = validation_one_epoch(val_loader, forward_model, device, args)

            # update DALI loader
            val_loader.reset()

            print(f"Accuracy of the network on the {args.val_files_list} val images: {test_stats['acc1']:.2f}%")
            
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(val_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(val_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(val_loss=test_stats['loss'], head="perf", step=epoch)
        
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in test_stats.items()}, 'epoch': epoch,
                        'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,
                        'n_parameters': n_parameters}
            
        if args.output_dir and is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"),mode="a",encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    print("start final_test")
    print("create test_loader start")
    test_loader = dali_dataloader(test_files_list,
                                args.dali_num_threads,
                                args.dali_py_num_workers // 2,
                                args.batch_size // 2,
                                input_size = args.input_size,
                                sequence_length = args.num_frames,
                                stride = args.sampling_rate,
                                use_sparse_sampling = args.sparse_sampling,
                                mode = "test",
                                seed = args.seed,
                                args = args)
    
    tta_nums = args.test_tta_num_segment * args.test_tta_num_crop
    args.test_steps_per_epoch = int((len(test_files_list) * tta_nums)  // args.world_size // (args.batch_size // 2))
    args.test_total_batch_size = args.world_size * (args.batch_size // 2)
    test_loader.step_data_num = args.test_steps_per_epoch
    args.test_files_list = len(test_files_list)
    print("create test_loader end")
    print("test videos: ", args.test_files_list)
    print("test tta videos: ", (len(test_files_list) * tta_nums))
    
    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    test_stats = final_test(test_loader, forward_model, device, preds_file, args)
    torch.distributed.barrier()
    if global_rank == 0:
        print("Start merging results...")
        final_top1, final_top5 = merge(args.output_dir, num_tasks = args.world_size)
        print(f"Accuracy of the network on the {args.test_files_list} test videos: \
            Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        log_stats = {'Final top-1': final_top1, 'Final Top-5': final_top5}
        if args.output_dir:
            with open(os.path.join(args.output_dir, "log.txt"), mode = "a", encoding = "utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, args):
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Val:'
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    for dali_batch in metric_logger.log_every(data_loader, 10, header):
        videos = dali_batch[0]
        target = dali_batch[1]
        videos = videos.to(device, non_blocking = True)
        target = target.to(device, non_blocking = True)
        target = target.view(-1)

        if args.enable_deepspeed:
            videos = videos.half()
            output = model(videos)
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = model(videos)
                loss = criterion(output, target)   

        acc1, acc5 = accuracy(output, target, topk = (1, 5))
        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n = batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n = batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks, method='prob'):
    assert method in ['prob', 'score']
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float32, sep=',')
            
            if name not in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
                
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            if method == 'prob':
                dict_feats[name].append(softmax(data))
            else:
                dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    p = Pool(32)
    # [pred, top1, top5, label]
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)
    return final_top1 * 100, final_top5 * 100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]


@torch.no_grad()
def final_test(data_loader, model, device, file, args):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    final_result = []
    for dali_batch in metric_logger.log_every(data_loader, 10, header):
        videos = dali_batch[0]; target = dali_batch[1]
        chunk_nb = dali_batch[2]; split_nb = dali_batch[3]; sample_idx = dali_batch[4]
        
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        chunk_nb = chunk_nb.to(device, non_blocking=True)
        split_nb = split_nb.to(device, non_blocking=True)
        sample_idx = sample_idx.to(device, non_blocking=True)
        
        target = target.view(-1); 
        chunk_nb = chunk_nb.view(-1); split_nb = split_nb.view(-1); sample_idx = sample_idx.view(-1)
        chunk_nb = chunk_nb.cpu().numpy().tolist()
        split_nb = split_nb.cpu().numpy().tolist()
        sample_idx = sample_idx.cpu().numpy().tolist()

        if args.enable_deepspeed:
            videos = videos.half()
            output = model(videos)
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = model(videos)
                loss = criterion(output, target)     

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(
                str(int(sample_idx[i])), str(output.data[i].float().cpu().numpy().tolist()),
                str(int(target[i].cpu().numpy())), str(int(chunk_nb[i])), str(int(split_nb[i])))
            final_result.append(string)
            
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
        
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)