from typing import Sequence

import torch
import numpy as np

def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    elif isinstance(data, dict):
        for key in data.keys():
            data[key] = to_tensor(data[key])
    else:
        raise TypeError(f'to_tensor : type {type(data)} cannot be converted to tensor.')

class ToTensor(object):
    def __init__(self, keys=None):
        self.keys = keys

    def __call__(self, sample):

        if self.keys is None:
            sample = to_tensor(sample)
        else:
            for key in keys:
                assert key in sample, \
                    f'ToTensor : not found key in sample : {key}'

            for key in self.keys:
                samples[key] = to_tensor(samples[key])

        return sample
        
class ImageToTensor(object):
    
    def __init__(self, keys: dict):
        self.keys = keys

    def transform(self, sample) -> dict:
        for key in keys:
            assert key in sample, \
                f'ImageToTensor : not found key in sample : {key}'
        
        for key in self.keys:
            img = sample[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            sample[key] = (ToTensor.to_tensor(img.transpose(2, 0, 1))).contiguous()

        return sample

class CollectKeys(object):
    def __init__(self, collect_keys):
        self.collect_keys = collect_keys

    def __call__(self, sample):
        for key in collect_keys:
            assert key in sample, \
                f'CollectKeys : not found key in sample : {key}'

        pack_data = {}
        for key in collect_keys:
            pack_data[key] = sample[key]

        return pack_data


