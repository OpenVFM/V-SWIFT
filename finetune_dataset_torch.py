import os
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import decord
from nvidia.dali.pipeline import Pipeline

import dataset_finetune.torch_video_transforms as video_transforms
from dataset_finetune.torch_random_erasing import RandomErasing
from torchvision import transforms


class DALIWarper(object):
    def __init__(self, dali_iter, step_data_num, mode="train"):
        self.iter = dali_iter
        self.step_data_num = step_data_num
        assert(mode in ["train", "val", "test"])
        self.mode = mode

    def __next__(self):
        if self.mode == "test":
            data_dict = self.iter.__next__()[0]
            videos = data_dict["videos"]
            labels = data_dict["labels"]
            chunk_nb = data_dict["chunk_nb"]
            split_nb = data_dict["split_nb"] 
            sample_idx = data_dict["sample_idx"]
            return videos, labels, chunk_nb, split_nb, sample_idx
        else:
            data_dict = self.iter.__next__()[0]
            videos = data_dict["videos"]
            labels = data_dict["labels"]
            return videos, labels

    def __iter__(self):
        return self

    def __len__(self):
        return self.step_data_num

    def reset(self):
        self.iter.reset()
    

class ExternalInputCallable:
    def __init__(self, mode, source_params):
        
        self.mode = mode
        assert(mode in ["train", "val", "test"])
        self.file_list = source_params['file_list']
        self.test_tta_num_segment = source_params['test_tta_num_segment']
        self.test_tta_num_crop = source_params['test_tta_num_crop']
                
        self.num_shards = source_params['num_shards']
        self.shard_id = source_params['shard_id']
        self.batch_size = source_params['batch_size']
        self.input_size = source_params['input_size']
        self.short_side_size = source_params['short_side_size']
        self.sequence_length = source_params['sequence_length']
        self.stride = source_params['stride']
        self.use_sparse_sampling = source_params['use_sparse_sampling']
        self.use_rgb = source_params['use_rgb']
        self.use_flip = source_params['use_flip']
        self.seed = source_params['seed']

        self.reprob = source_params['reprob']
        self.remode = source_params['remode']
        self.recount = source_params['recount']
        self.resplit = source_params['resplit']
        self.aa = source_params['aa']
        self.train_interpolation = source_params['train_interpolation']
        self.aug_transform = video_transforms.create_random_augment(
            input_size = (self.input_size, self.input_size),
            auto_augment = self.aa,
            interpolation = self.train_interpolation,
        )
        if self.reprob > 0:
            self.erase_transform = RandomErasing(
                self.reprob,
                mode = self.remode,
                max_count = self.recount,
                num_splits = self.recount,
                device = "cpu",
            )
        self.data_resize = video_transforms.Compose([
                                        video_transforms.Resize(size=(self.short_side_size), 
                                                                interpolation='bilinear')])
  
        # If the dataset size is not divisible by number of shards, the trailing samples will be omitted.
        self.shard_size = len(self.file_list) // self.num_shards
        self.shard_offset = self.shard_size * self.shard_id
        # drop last batch
        self.full_iterations = self.shard_size // self.batch_size
        # so that we don't have to recompute the `self.perm` for every sample
        self.perm = None
        self.last_seen_epoch = None
        self.replace_example_info = self.file_list[0]


    def sparse_sampling_get_frameid_data(self, video_path, sequence_length, test_info):
        decord_vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
        duration = len(decord_vr)

        if self.mode == "train" or self.mode == "val":
            average_duration = duration // sequence_length
            all_index = []
            if average_duration > 0:
                if self.mode == 'val':
                    all_index = list(
                        np.multiply(list(range(sequence_length)), average_duration) +
                        np.ones(sequence_length, dtype = int) * (average_duration // 2))
                else:
                    all_index = list(
                        np.multiply(list(range(sequence_length)), average_duration) +
                        np.random.randint(average_duration, size = sequence_length))
            elif duration > sequence_length:
                if self.mode == 'val':
                    all_index = list(range(sequence_length))
                else:
                    all_index = list(np.sort(np.random.randint(duration, 
                                                               size = sequence_length)))
            else:
                all_index = [0] * (sequence_length - duration) + list(range(duration))
            frame_id_list = list(np.array(all_index))
            
            decord_vr.seek(0)
            video_data = decord_vr.get_batch(frame_id_list).asnumpy()            
            if self.use_rgb:
                video_data = video_data[:,:,:,::-1]
            return video_data
        else:
            chunk_nb, split_nb, video_idx = test_info
            tick = duration / float(sequence_length)
            all_index = []
            for t_seg in range(self.test_tta_num_segment):
                tmp_index = [
                    int(t_seg * tick / self.test_tta_num_segment + tick * x)
                    for x in range(sequence_length)
                ]
                all_index.extend(tmp_index)
            all_index = list(np.sort(np.array(all_index)))
            cur_index = all_index[chunk_nb::self.test_tta_num_segment]     
            decord_vr.seek(0)
            video_data = decord_vr.get_batch(cur_index).asnumpy()     
            if self.use_rgb:
                video_data = video_data[:,:,:,::-1]

            # check torch_test
            video_data = self.data_resize(video_data)
            if isinstance(video_data, list):
                video_data = np.stack(video_data, 0)


            vf, vh, vw, vc = video_data.shape
            short_side_size = min(vh, vw)
            long_side_size = max(vh, vw)
            spatial_step = 1.0 * (long_side_size - short_side_size) / (self.test_tta_num_crop - 1)
            spatial_start = int(split_nb * spatial_step)
            if vh >= vw:
                video_data = video_data[:, spatial_start:spatial_start + short_side_size, :, :]
            else:
                video_data = video_data[:, :, spatial_start:spatial_start + short_side_size, :]
            return video_data


    def dense_sampling_get_frameid_data(self, video_path, sequence_length, stride, test_info):
        decord_vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
        duration = len(decord_vr)

        if self.mode == "train" or self.mode == "val":
            pass
        else:
            pass


    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
        inverse_uniform_sampling=False,
        aspect_ratio=None,
        scale=None,
        motion_shift=False,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
            inverse_uniform_sampling (bool): if True, sample uniformly in
                [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
                scale. If False, take a uniform sample from [min_scale,
                max_scale].
            aspect_ratio (list): Aspect ratio range for resizing.
            scale (list): Scale range for resizing.
            motion_shift (bool): Whether to apply motion shift for resizing.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            if aspect_ratio is None and scale is None:
                frames, _ = video_transforms.random_short_side_scale_jitter(
                    images=frames,
                    min_size=min_scale,
                    max_size=max_scale,
                    inverse_uniform_sampling=inverse_uniform_sampling,
                )
                frames, _ = video_transforms.random_crop(frames, crop_size)
            else:
                transform_func = (
                    video_transforms.random_resized_crop_with_shift
                    if motion_shift else video_transforms.random_resized_crop)
                frames = transform_func(
                    images=frames,
                    target_height=crop_size,
                    target_width=crop_size,
                    scale=scale,
                    ratio=aspect_ratio,
                )
            if random_horizontal_flip:
                frames, _ = video_transforms.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = video_transforms.random_short_side_scale_jitter(
                frames, min_scale, max_scale)
            frames, _ = video_transforms.uniform_crop(frames, crop_size,
                                                    spatial_idx)
        return frames


    def tensor_normalize(self, tensor, mean, std):
        """
        Normalize a given tensor by subtracting the mean and dividing the std.
        Args:
            tensor (tensor): tensor to normalize.
            mean (tensor or list): mean value to subtract.
            std (tensor or list): std to divide.
        """
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()
            tensor = tensor / 255.0
        if type(mean) == list:
            mean = torch.tensor(mean)
        if type(std) == list:
            std = torch.tensor(std)
        tensor = tensor - mean
        tensor = tensor / std
        return tensor


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
        if self.mode == "test":
            test_info = example_info[-3:]
            example_info = example_info[:-3]
        else:
            test_info = None

        if len(example_info) == 4:
            video_path, video_label, sframe, eframe = example_info
        elif len(example_info) == 6:
            source_name, video_path, video_label, sframe, eframe, frame_nums = example_info
        else:
            print("format: video_path,video_label")
            print("format: video_path video_label sframe eframe")
            print("format: source_name video_path video_label sframe eframe frame_nums")
            exit(1)
            
            
        try:
            if self.use_sparse_sampling:
                video_data = self.sparse_sampling_get_frameid_data(video_path, self.sequence_length, test_info)
            else:
                video_data = self.dense_sampling_get_frameid_data(video_path, self.sequence_length, self.stride, test_info)
        except:
            print("error", video_path)
            _, video_path, video_label, _, _, _ = self.replace_example_info

            if self.use_sparse_sampling:
                video_data = self.sparse_sampling_get_frameid_data(video_path, self.sequence_length, test_info)
            else:
                video_data = self.dense_sampling_get_frameid_data(video_path, self.sequence_length, self.stride, test_info)
        
        if self.mode == "train":
            buffer = video_data
            buffer = [transforms.ToPILImage()(frame) for frame in buffer]
            buffer = self.aug_transform(buffer)
            buffer = [transforms.ToTensor()(img) for img in buffer]
            buffer = torch.stack(buffer)  # T C H W
            buffer = buffer.permute(0, 2, 3, 1)  # T H W C
            # T H W C
            buffer = self.tensor_normalize(buffer, 
                                      [0.485, 0.456, 0.406], 
                                      [0.229, 0.224, 0.225])
            # T H W C -> C T H W.
            buffer = buffer.permute(3, 0, 1, 2)

            scl, asp = (
                [0.08, 1.0],
                [0.75, 1.3333],
            )
            buffer = self.spatial_sampling(
                buffer,
                spatial_idx = -1,
                min_scale = 256,
                max_scale = 320,
                crop_size = self.input_size,
                random_horizontal_flip = self.use_flip,
                inverse_uniform_sampling = False,
                aspect_ratio = asp,
                scale = scl,
                motion_shift = False)
            if self.reprob > 0:
                buffer = buffer.permute(1, 0, 2, 3)  # C T H W -> T C H W
                buffer = self.erase_transform(buffer)
                buffer = buffer.permute(1, 0, 2, 3)  # T C H W -> C T H W
            video_data = buffer.numpy()    

        if self.mode == "test":
            chunk_nb, split_nb, video_idx = test_info
            return video_data, np.int64([int(video_label)]), \
                np.int64([int(chunk_nb)]), np.int64([int(split_nb)]), np.int64([int(video_idx)])
        else:
            return video_data, np.int64([int(video_label)])
    
    
def dali_dataloader(file_list,
                    dali_num_threads,
                    dali_py_num_workers,
                    batch_size,
                    input_size = 224,
                    sequence_length = 16,
                    stride = 4,
                    use_sparse_sampling = False, 
                    mode = "train",
                    seed = 0,
                    args = None):

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    if mode == "test":
        test_file_list = []
        for chunk_nb in range(args.test_tta_num_segment):
            for split_nb in range(args.test_tta_num_crop):
                for video_idx,  video_sample in enumerate(file_list):
                    new_sample = video_sample[:] + [chunk_nb, split_nb, video_idx]
                    test_file_list.append(new_sample)
        file_list = test_file_list

    source_params = {
        "batch_size": batch_size,
        "seed": seed + rank,
        "num_shards": world_size, 
        "shard_id": rank,
        "file_list": file_list,
        "input_size": input_size,
        "short_side_size": args.short_side_size,
        "sequence_length": sequence_length,
        "stride": stride,
        "use_sparse_sampling": use_sparse_sampling,
        "use_rgb": args.use_decord_bgr,
        "use_flip": args.use_random_horizontal_flip,
        "mean": args.mean,
        "std": args.std,
        "reprob": args.reprob,
        "remode": args.remode,
        "recount": args.recount,
        "resplit": args.resplit,
        "aa": args.aa,
        "train_interpolation": args.train_interpolation,
        "test_tta_num_segment": args.test_tta_num_segment,
        "test_tta_num_crop": args.test_tta_num_crop,
    }
    
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
        if mode == "train":
            videos, labels = fn.external_source(
                source = ExternalInputCallable(mode, source_params),
                num_outputs = 2,
                batch = False,
                parallel = True,
                dtype = [types.FLOAT, types.INT64],
                layout = ["FHWC", "C"]
            )
            videos = videos.gpu()
            labels = labels.gpu()
            pipe.set_outputs(videos, labels)
        elif mode == "val":
            videos, labels = fn.external_source(
                source = ExternalInputCallable(mode, source_params),
                num_outputs = 2,
                batch = False,
                parallel = True,
                dtype = [types.UINT8, types.INT64],
                layout = ["FHWC", "C"]
            )
            videos = videos.gpu()
            videos = fn.resize(videos, device = "gpu", antialias = True, interp_type = types.INTERP_LINEAR,
                            resize_shorter = source_params['short_side_size'])
            videos = fn.crop(videos, device = "gpu", crop = [source_params['input_size'], source_params['input_size']])
            videos = fn.crop_mirror_normalize(videos, device = "gpu", dtype = types.FLOAT, output_layout = "CFHW",
                                            mean = [m*255.0 for m in source_params['mean']],
                                            std = [m*255.0 for m in source_params['std']])
            labels = labels.gpu()
            pipe.set_outputs(videos, labels)
        else:
            videos, labels, chunk_nb, split_nb, video_idx = fn.external_source(
                source = ExternalInputCallable(mode, source_params),
                num_outputs = 5,
                batch = False,
                parallel = True,
                dtype = [types.UINT8, types.INT64, types.INT64, types.INT64, types.INT64],
                layout = ["FHWC", "C", "C", "C", "C"]
            )
            videos = videos.gpu()
            videos = fn.resize(videos, device = "gpu", antialias = True, interp_type = types.INTERP_LINEAR,
                            resize_y = source_params['input_size'], resize_x = source_params['input_size'])
            videos = fn.crop_mirror_normalize(videos, device = "gpu", dtype = types.FLOAT, output_layout = "CFHW",
                                            mean = [m*255.0 for m in source_params['mean']],
                                            std = [m*255.0 for m in source_params['std']])
            labels = labels.gpu(); chunk_nb = chunk_nb.gpu(); split_nb = split_nb.gpu(); video_idx = video_idx.gpu()
            pipe.set_outputs(videos, labels, chunk_nb, split_nb, video_idx)
    pipe.build()


    dataloader = DALIWarper(
        dali_iter = DALIGenericIterator(pipelines=pipe,
            output_map=['videos', 'labels', 'chunk_nb', 'split_nb', 'sample_idx'] \
                        if mode == "test" else ['videos', 'labels'],
            auto_reset=False,
            size=-1,
            last_batch_padded=False,
            last_batch_policy=LastBatchPolicy.FILL,
            prepare_first_batch=False),
        step_data_num = len(file_list) // world_size // batch_size,
        mode = mode
    )
    return dataloader