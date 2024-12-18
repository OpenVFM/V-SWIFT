import os
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import decord
import random


class DALIWarper(object):
    def __init__(self, dali_iter, step_data_num):
        self.iter = dali_iter
        self.step_data_num = step_data_num

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        videos = data_dict["videos"].cuda()
        labels = data_dict["labels"].cuda()
        return videos, labels

    def __iter__(self):
        return self

    def __len__(self):
        return self.step_data_num

    def reset(self):
        self.iter.reset()


@torch.no_grad()
class SyntheticWarper(object):
    def __init__(self, batch_size, input_size, sequence_length, local_rank, set_max_sample=160000):
        
        self.set_max_sample = int(set_max_sample)
        
        videos = torch.randint(
            low=0,
            high=255,
            size=(batch_size, 3, sequence_length, input_size, input_size),
            dtype=torch.float32,
            device=local_rank,
        )
        videos = videos / 255.0
        
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        mean = torch.as_tensor(mean, dtype=torch.float32, device=local_rank)
        std = torch.as_tensor(std, dtype=torch.float32, device=local_rank)
        videos.sub_(mean[None, :, None, None, None]).div_(std[None, :, None, None, None])
        labels = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=local_rank)
        
        self.videos = videos
        self.labels = labels

    def __next__(self):
        return self.videos, self.labels

    def __iter__(self):
        return self

    def __len__(self):
        return self.set_max_sample

    def reset(self):
        return


class VideoMultiScaleCrop(object):

    def __init__(self,
                 input_size_h=224,
                 input_size_w=224,
                 scales=[1, .875, .75, .66],
                 max_distort=1,
                 fix_crop=True,
                 more_fix_crop=True):

        self.input_size_h = input_size_h
        self.input_size_w = input_size_w
        self.scales = scales

        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop


    def __call__(self, video_h, video_w):
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(image_w=video_w, 
                                                                    image_h=video_h)
        
        return crop_w, crop_h, offset_w, offset_h

    def _sample_crop_size(self, image_w, image_h):

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * temp) for temp in self.scales]
        crop_h = [
            self.input_size_h if abs(temp - self.input_size_h) < 3 else temp
            for temp in crop_sizes
        ]
        crop_w = [
            self.input_size_w if abs(temp - self.input_size_w) < 3 else temp
            for temp in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h,
                                       crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret
    

class ExternalInputCallable:
    def __init__(self, source_params):

        self.file_list = source_params['file_list']
        self.num_shards = source_params['num_shards']
        self.shard_id = source_params['shard_id']
        self.batch_size = source_params['batch_size']
        self.input_size = source_params['input_size']
        self.sequence_length = source_params['sequence_length']
        self.stride = source_params['stride']
        self.use_rgb = source_params['use_rgb']
        self.seed = source_params['seed']

        self.msc_crop = VideoMultiScaleCrop(input_size_h=self.input_size,
                                        input_size_w=self.input_size,
                                        scales=[1, 0.875, 0.75, 0.66])

        # If the dataset size is not divisible by number of shards, the trailing samples will be omitted.
        self.shard_size = len(self.file_list) // self.num_shards
        self.shard_offset = self.shard_size * self.shard_id
        # drop last batch
        self.full_iterations = self.shard_size // self.batch_size
        # so that we don't have to recompute the `self.perm` for every sample
        self.perm = None
        self.last_seen_epoch = None
        self.replace_example_info = self.file_list[0]


    def get_frame_id_list(self, video_path, sequence_length, stride, temporal_jitter=False):
        decord_vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
        duration = len(decord_vr)

        skip_length = sequence_length * stride
        average_duration = (duration - skip_length  + 1)
        if average_duration > 1:
            #randint: [low, high)
            offset = np.random.randint(average_duration, size=1)[0]

            if temporal_jitter:
                skip_offsets = np.random.randint(stride, size=sequence_length)
            else:
                skip_offsets = np.zeros(sequence_length, dtype=int)

            frame_id_list = []
            for i, _ in enumerate(range(0, skip_length, stride)):
                if offset + skip_offsets[i] < duration:
                    frame_id = offset + skip_offsets[i]
                else:
                    frame_id = offset
                frame_id_list.append(frame_id)
                if offset + stride < duration:
                    offset += stride

            if len(frame_id_list) < sequence_length:
                frame_id_list = [0] * (sequence_length - len(frame_id_list)) + frame_id_list
        elif duration > sequence_length:
            frame_id_list = list(np.sort(np.random.randint(duration, size=sequence_length)))
        else:
            frame_id_list = [0] * (sequence_length - duration) + list(range(duration))

        decord_vr.seek(0)
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
        
        if self.use_rgb:
            video_data = video_data[:,:,:,::-1]

        vf, vh, vw, vc = video_data.shape
        crop_w, crop_h, offset_w, offset_h = self.msc_crop(video_h=vh, video_w=vw)
        video_data = video_data[:, offset_h:offset_h+crop_h, offset_w:offset_w+crop_w, :]
        return video_data
        

    def __call__(self, sample_info):
        #sample_info
        #idx_in_epoch – 0-based index of the sample within epoch
        #idx_in_batch – 0-based index of the sample within batch
        #iteration – number of current batch within epoch
        #epoch_idx – number of current epoch
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration

        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            cur_seed = self.seed + sample_info.epoch_idx
            self.perm = np.random.default_rng(seed=cur_seed).permutation(len(self.file_list))
            
        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]

        example_info = self.file_list[sample_idx]

        # check
        if len(example_info) == 6:
            source_name, video_path, video_label, sframe, eframe, frame_nums = example_info
        else:
            print("format: video_path,video_label")
            print("format: video_path video_label sframe eframe")
            print("format: source_name video_path video_label sframe eframe frame_nums")
            exit(1)
            
        try:
            video_data = self.get_frame_id_list(video_path, self.sequence_length, self.stride)
        except:
            print("error", video_path)
            _, video_path, video_label, _, _, _ = self.replace_example_info
            video_data = self.get_frame_id_list(video_path, self.sequence_length, self.stride)

        return video_data, np.int64([int(video_label)])


def dali_dataloader(file_list,
                    dali_num_threads,
                    dali_py_num_workers,
                    batch_size,
                    gpus_not_equal_num_shards = False,
                    input_size = 224,
                    sequence_length = 16,
                    stride = 4,
                    gpus_per_node = 8,
                    use_rgb = False,
                    use_flip = False,
                    use_synthetic = False,
                    set_max_sample = 160000,
                    mean = [0.48145466, 0.4578275, 0.40821073],
                    std = [0.26862954, 0.26130258, 0.27577711],
                    seed = 0):

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    if use_synthetic:
        return SyntheticWarper(batch_size, input_size, sequence_length, local_rank, set_max_sample)

    source_params = {
        "file_list": file_list,
        "batch_size": batch_size,
        "input_size": input_size,
        "sequence_length": sequence_length,
        "stride": stride,
        "seed": seed + rank,
        "use_rgb": use_rgb,
    }
    
    if gpus_not_equal_num_shards == True:
        source_params.update({"num_shards": gpus_per_node, "shard_id": local_rank}) 
    else:
        source_params.update({"num_shards": world_size, "shard_id": rank}) 

    pipe = Pipeline(
        batch_size = batch_size,
        num_threads = dali_num_threads,
        device_id = local_rank,
        seed = seed + rank,
        py_num_workers = dali_py_num_workers,
        py_start_method = 'spawn',
        prefetch_queue_depth = 1,
    )

    with pipe:
        videos, labels = fn.external_source(
            source=ExternalInputCallable(source_params),
            num_outputs=2,
            batch=False,
            parallel=True,
            dtype=[types.UINT8, types.INT64],
            layout = ["FHWC", "C"])
        videos = videos.gpu()
        videos = fn.resize(videos, device = "gpu", dtype = types.FLOAT,
                        antialias = True, interp_type = types.INTERP_LINEAR,
                        resize_y = input_size, resize_x = input_size)
        if use_flip:
            videos = fn.flip(videos, device = "gpu", horizontal = fn.random.coin_flip(probability=0.5))
        videos = fn.crop_mirror_normalize(videos, device = "gpu", 
                                        dtype = types.FLOAT, output_layout = "CFHW",
                                        mean = [m*255.0 for m in mean], std = [m*255.0 for m in std])
        labels = labels.gpu()
        pipe.set_outputs(videos, labels)
    pipe.build()

    dataloader = DALIWarper(
        dali_iter = DALIGenericIterator(pipelines=pipe,
            output_map=['videos', 'labels'],
            auto_reset=False,
            size=-1,
            last_batch_padded=False,
            last_batch_policy=LastBatchPolicy.FILL,
            prepare_first_batch=False),
        step_data_num = len(file_list) // batch_size // world_size,
    )
    
    return dataloader