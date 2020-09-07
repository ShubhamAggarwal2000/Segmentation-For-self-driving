import sys
sys.path.append('.')

import torch
from MML.models.drunet import DRUNet

def test_model():
    drunet = DRUNet(3, 3)

    sample_img = torch.randn((1, 3, 224, 224))
    print(drunet(sample_img).shape)
