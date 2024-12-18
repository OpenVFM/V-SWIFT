import os
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali.auto_aug import rand_augment
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.pipeline import Pipeline

import decord
import torch
import numpy as np

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

        # check
        if len(example_info) == 6:
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
                                                                  
        # return
        if self.mode == "test":
            chunk_nb, split_nb, video_idx = test_info
            return video_data, np.int64([int(video_label)]), \
                np.int64([int(chunk_nb)]), np.int64([int(split_nb)]), np.int64([int(video_idx)])
        else:
            return video_data, np.int64([int(video_label)])


@pipeline_def(enable_conditionals=True)
def dali_pipeline(mode, source_params):
    if mode == "train":
        videos, labels = fn.external_source(
            source = ExternalInputCallable(mode, source_params),
            num_outputs = 2,
            batch = False,
            parallel = True,
            dtype = [types.UINT8, types.INT64],
            layout = ["FHWC", "C"]
        )
        videos = videos.gpu()
        videos = rand_augment.rand_augment(videos, n = 4, m = 7, fill_value = 128, monotonic_mag = True)
        videos = fn.random_resized_crop(videos, random_area = (0.08, 1.0), random_aspect_ratio = (0.75, 1.3333),
                                        size = [source_params['input_size'], source_params['input_size']],
                                        num_attempts = 10, antialias = True, interp_type = types.INTERP_LINEAR)
        if source_params['reprob'] > 0:
            erase_probability = fn.random.coin_flip(dtype=types.BOOL, probability=source_params['reprob'])
            if erase_probability:
                mask = videos * 0
                # anchor=(y0, x0, y1, x1, …);shape=(h0, w0, h1, w1, …)
                mask = fn.erase(mask, device = "gpu", axis_names = "HW", fill_value = 255,
                        anchor = fn.random.uniform(range=(0, source_params['input_size']), shape=(2, )),
                        shape = fn.random.uniform(range=(20, 90), shape=(2, )))
                noise = fn.random.normal(videos, device = "gpu", dtype = types.INT8)
                videos = (videos & (255 - mask)) | (noise & mask)
            else:
                # align dali-types
                mask = videos * 0
                videos = (videos & (255 + mask))
                
        if source_params['use_flip']:
            videos = fn.flip(videos, device = "gpu", horizontal = fn.random.coin_flip(probability = 0.5))

        videos = fn.crop_mirror_normalize(videos, device = "gpu", dtype = types.FLOAT, output_layout = "CFHW",
                                        mean = [m*255.0 for m in source_params['mean']],
                                        std = [m*255.0 for m in source_params['std']])
        labels = labels.gpu()
        return videos, labels
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
        return videos, labels
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
        return videos, labels, chunk_nb, split_nb, video_idx

    
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
        "test_tta_num_segment": args.test_tta_num_segment,
        "test_tta_num_crop": args.test_tta_num_crop,
    }
    
    pipe = dali_pipeline(
        batch_size = batch_size,
        num_threads = dali_num_threads,
        device_id = local_rank,
        seed = seed + rank,
        py_num_workers = dali_py_num_workers,
        py_start_method = 'spawn',
        prefetch_queue_depth = 1,
        mode = mode,
        source_params = source_params
    )
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