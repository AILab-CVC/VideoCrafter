import os
import random
import bisect

import pandas as pd

import omegaconf
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from decord import VideoReader, cpu
import torchvision.transforms._transforms_video as transforms_video

class WebVid(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fps_schedule=None,
                 fs_probs=None,
                 bs_per_gpu=None,
                 trigger_word='',
                 dataname='',
                 ):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.frame_stride = frame_stride
        self.fps_max = fps_max
        self.load_raw_resolution = load_raw_resolution
        self.fs_probs = fs_probs
        self.trigger_word = trigger_word
        self.dataname = dataname

        self._load_metadata()
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms_video.RandomCropVideo(crop_resolution)
            elif spatial_transform == "resize_center_crop":
                assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(resolution),
                    transforms_video.CenterCropVideo(resolution),
                    ])
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None
        
        self.fps_schedule = fps_schedule
        self.bs_per_gpu = bs_per_gpu
        if self.fps_schedule is not None:
            assert(self.bs_per_gpu is not None)
            self.counter = 0
            self.stage_idx = 0
        
    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path)
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]
    
    def _get_video_path(self, sample):
        if self.dataname == "loradata":
            rel_video_fp =  str(sample['videoid']) + '.mp4'
            full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        else:
            rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
            full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp, rel_video_fp
    
    def get_fs_based_on_schedule(self, frame_strides, schedule):
        assert(len(frame_strides) == len(schedule) + 1) # nstage=len_fps_schedule + 1
        global_step = self.counter // self.bs_per_gpu # TODO: support resume.
        stage_idx = bisect.bisect(schedule, global_step)
        frame_stride = frame_strides[stage_idx]
        # log stage change
        if stage_idx != self.stage_idx:
            print(f'fps stage: {stage_idx} start ... new frame stride = {frame_stride}')
            self.stage_idx = stage_idx
        return frame_stride
    
    def get_fs_based_on_probs(self, frame_strides, probs):
        assert(len(frame_strides) == len(probs))
        return random.choices(frame_strides, weights=probs)[0]

    def get_fs_randomly(self, frame_strides):
        return random.choice(frame_strides)
    
    def __getitem__(self, index):
        
        if isinstance(self.frame_stride, list) or isinstance(self.frame_stride, omegaconf.listconfig.ListConfig):
            if self.fps_schedule is not None:
                frame_stride = self.get_fs_based_on_schedule(self.frame_stride, self.fps_schedule)
            elif self.fs_probs is not None:
                frame_stride = self.get_fs_based_on_probs(self.frame_stride, self.fs_probs)
            else:
                frame_stride = self.get_fs_randomly(self.frame_stride)
        else:
            frame_stride = self.frame_stride
        assert(isinstance(frame_stride, int)), type(frame_stride)

        while True:
            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            video_path, rel_fp = self._get_video_path(sample)
            caption = sample['caption']+self.trigger_word
            
            # make reader
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
                if len(video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            # sample strided frames
            all_frames = list(range(0, len(video_reader), frame_stride))
            if len(all_frames) < self.video_length: # recal a max fs
                frame_stride = len(video_reader) // self.video_length
                assert(frame_stride != 0)
                all_frames = list(range(0, len(video_reader), frame_stride))

            # select a random clip
            rand_idx = random.randint(0, len(all_frames) - self.video_length)
            frame_indices = all_frames[rand_idx:rand_idx+self.video_length]
            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}")
                index += 1
                continue

        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        if self.resolution is not None:
            assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        
        fps_ori = video_reader.get_avg_fps()
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        
        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride}

        if self.fps_schedule is not None:
            self.counter += 1
        return data
    
    def __len__(self):
        return len(self.metadata)
