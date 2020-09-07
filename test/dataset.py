import sys
sys.path.append('.')
import matplotlib.pyplot as plt

import torch
from MML.datasets.hc18 import TrainDataset
from MML.preprocessing import Transform
from MML.visualization import visualize_dataset


def test_dataset():

    transform_dict = {
        # Resize
        'img_size': (272, 400),

        # Zoom In
        'zoom_in_prob': 0.5,
        'zoom_in_scale': (1, 1.1),

        # Zoom Out
        'zoom_out_prob': 0.5,
        'zoom_out_scale': (1, 1.1),
        'fill_mean': True,

        # Select Both zoom in and zoom out
        'zoom_in_out': True,

        # Flip  
        'h_flip': True,
        'v_flip': False,

        # Rotation
        'angle': 5,

        # gamma transform (intensity scaling)
        'gamma': (0.8, 1.1),

        # additive noise
        'additive_noise': 0.1,

        # Normalization
        'mean': [0.5],
        'std': [0.5],
    }

    transform = Transform(transform_dict)
    dataset = TrainDataset(transform)
    visualize_dataset(dataset)