import torch
import torch.utils.data as data
from PIL import Image
import json
import os
import copy
from transform_utils import *

def get_ipn_data(config, split, spatial_transform, temporal_transform):
    return IPN_Dataset(split, config.video_path, config.annotation_path, 
                       spatial_transform=spatial_transform,
                       temporal_transform=temporal_transform)

class IPN_Dataset(data.Dataset):
    def __init__(self, split, video_pth, annotation_pth, n_samples_per_video=1, 
                 spatial_transform=None, temporal_transform=None, sample_duration=32):
        self.init_data(video_pth, annotation_pth, split, n_samples_per_video, sample_duration)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.sample_duration = sample_duration

    def init_data(self, root_pth, annotation_pth, split, n_samples_per_video, sample_duration):
        with open(annotation_pth, 'r') as annotation_file:
            annotation_data = json.load(annotation_file)
        
        video_names = []
        annotations = []

        for k, v in annotation_data['database'].items():
            subset = v['subset']
            if subset == split:
                label = v['annotations']['label']
                video_names.append(k.split('^')[0])
                annotations.append(v['annotations'])
        
        class_to_idx = {}
        idx_to_class = {}
        idx = 0

        for class_label in annotation_data['labels']:
            class_to_idx[class_label] = idx
            idx += 1

        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        dataset = []

        for i in range(len(video_names)):
            video_path = os.path.join(root_pth, video_names[i])
            
            if not os.path.exists(video_path):
                continue

            begin_t = int(annotations[i]['start_frame'])
            end_t = int(annotations[i]['end_frame'])
            n_frames = end_t - begin_t + 1
            sample = {
                'video': video_path,
                'segment': [begin_t, end_t],
                'n_frames': n_frames,
                'video_id': i
            }

            if len(annotations) != 0:
                sample['label'] = class_to_idx[annotations[i]['label']]
            else:
                sample['label'] = -1

            if n_samples_per_video == 1:
                sample['frame_indices'] = list(range(begin_t, end_t + 1))
                dataset.append(sample)
            else:
                if n_samples_per_video > 1:
                    step = max(1,
                            math.ceil((n_frames - 1 - sample_duration) /
                                        (n_samples_per_video - 1)))
                else:
                    step = sample_duration
                for j in range(1, n_frames, step):
                    sample_j = copy.deepcopy(sample)
                    sample_j['frame_indices'] = list(
                        range(j, min(n_frames + 1, j + sample_duration)))
                    dataset.append(sample_j)
        
        self.data = dataset
        self.idx_to_class = idx_to_class

    def load_clip(self, video_dir_pth, frame_idx, sample_duration):
        video = []
        
        for i in frame_idx:
            img_pth = os.path.join(video_dir_pth, '{:s}_{:06d}.jpg'.format(video_dir_pth.split('/')[-1],i))
            if os.path.exists(img_pth):
                with open(img_pth, 'rb') as f:
                    with Image.open(f) as img:
                        img.convert('RGB')
                        video.append(img)
            else:
                print(img_pth, " does not exist")
                return video
        
        return video

    def __getitem__(self, idx):
        path = self.data[idx]['video']
        frame_idx = self.data[idx]['frame_indices']

        if self.temporal_transform is not None:
            frame_idx = self.temporal_transform(frame_idx)

        clip = self.load_clip(path, frame_idx, self.sample_duration)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
    
        im_dim = clip[0].size()[-2:]
        clip = torch.cat(clip, 0).view((self.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
        target = self.data[idx]['label']

        return clip, target

    def __len__(self):
        return len(self.data)