import sys
sys.path.append('.')

import torch
import torch.optim as optim

from MML.models.drunet import DRUNet
from MML.datasets.hc18 import TrainDataset
from MML.utils.dataloader import generate_dataloaders
from MML.helper_functions import fit_network
from MML.loss import BCELoss
from MML.metrics import find_acc_binary

def test_training():
    criterion = BCELoss()
    network = DRUNet(1,1)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1)

    log_dict={
        'name': 'Package',
        'entity': 'origin_health',
        'project': 'package_test',
        'log_histograms': False
        }

    train_dataset = TrainDataset()
    train_loader, valid_loader = generate_dataloaders(train_dataset, batch_size=4, valid_split=0.2, num_workers=4)

    fit_network(network, 
                train_loader, 
                valid_loader, 
                criterion, 
                find_acc_binary, 
                optimizer, 
                num_epochs=100, 
                checkpoint_path=None, 
                start_from_checkpoint=False, 
                log_dict=None, 
                lr_scheduler=lr_scheduler)