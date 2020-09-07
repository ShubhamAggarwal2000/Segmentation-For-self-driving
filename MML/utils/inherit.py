import sys
sys.path.append('.')

import torch
from torchsummary import summary as torch_summary

from MML.helper_functions import fit_network
from MML.utils.utils import load_network
from MML.utils.dataloader import generate_dataloaders
from MML.visualization import visualize_dataset, visualize_prediction, visualize_wandb


class DatasetInherit():
    """ Class to inherit special methods into the Dataset Class """
    def __init__(self):
        pass

    def visualize(self, index=None, alpha=0.1, denormalize=True, colorbar=False):
        """
        Dataset method that calls visualize_dataset() on the dataset object. 

        Parameters
        ----------
        index : int, optional
            index of any specific data that needs to be visualized, 
            If not specified it displays random data from the dataset, by default None
        alpha : float, optional
            opaqueness value, higher the value more opaque will be 
            the mask imposition on image, by default 0.1
        denormalize : boolean value, True by default
            whether we want denormalized image or not
        colorbar : whether we want to display colorbar, 
        by default True
        """
        transform = self.custom_transform if denormalize else None
        visualize_dataset(self, index=index, alpha=alpha, 
                    transform=transform, colorbar=colorbar)

    def batch(self, batch_size=1, valid_split=0.1, test_split=0.0, num_workers=0):
        """
        Dataset method that calls generate_dataloaders() on the dataset object.
        Generates dataloaders with specified batch size.
        Performs random train-valid-test split.

        Parameters
        ----------
            batch_size : int, optional
                batch size (will be same for train, valid and test loaders), by default 1
            valid_split : float, optional
                fraction of train set used for validation (between 0 and 1), by default 0.1
            test_split : float, optional
                fraction of train set used for test [0 implies only train and valid loaders
                are generated](between 0 and 1), by default 0.0
            num_workers : int, optional
                no of CPU threads (preferrably between 0 and 4), by default 0

        Returns
        -------
        torch dataloader objects
            train_loader, valid_loader if test_split = 0
            train_loader, valid_loader, test_loader if test_split != 0
        """

        *loader, = generate_dataloaders(
                    self, 
                    batch_size=batch_size, 
                    valid_split=valid_split, 
                    test_split=test_split, 
                    num_workers=num_workers)
        return loader

class NetworkInherit():
    """ Class to inherit special methods into the Network Class """
    def __init__(self):
        pass

    def predict(self, dataset, index=None, alpha=0.1, 
                quantize=True, denormalize=True, colorbar=False):
        """
        network method that calls visualize_prediction() on the network object. 
        Used for visualizing the network predicted mask.

        Parameters
        ----------
        dataset : torch dataset
            the dataset to be visualized
        index : int, optional
            index of any specific data that needs to be visualized, 
            If not specified it displays random data from the dataset, by default None
        alpha : float, optional
            opaqueness value, higher the value more opaque will be 
            the mask imposition on image, by default 0.1
        quantize : boolean, optional
            Whether you want the mask to be quantized
        denormalize : boolean value, True by default
            whether we want denormalized image or not
        colorbar : whether we want to display colorbar, 
            by default True
        """

        if denormalize:
            mean = dataset.mean
            std = dataset.std
        else:
            mean = None
            std = None

        visualize_prediction(self, dataset, index=index, alpha=alpha, 
                            quantize=quantize, mean=mean, std=std, colorbar=colorbar)

    def predict_wandb(self, dataset, title="Head Segmentation"):
        mean = dataset.mean
        std = dataset.std

        visualize_wandb(self, dataset, mean=mean, std=std, title=title)


    def fit(self, train_loader, valid_loader, criterion, acc_metric, 
                optimizer, num_epochs=5, checkpoint_path=None, metric='loss',
                start_from_checkpoint=False, lr_scheduler=None, log_dict={}):

        """
        Function that calls fit_network() on the network object. 
        Trains and validates the given network for specified number of epochs. 
        Includes logging, saving checkpoints and lr_scheduling facilities.

        Parameters
        ----------
        train_loader : torch dataloader
            provides batches of training data
        vaild_loader : torch dataloader
            provides batches of validation data
        criterion : nn.Module
            loss criterion
        acc_metric : function
            accuracy metric
        optimizer : torch.optim
            optimizer
        num_epochs : int, optional
            number of epochs, by default 5
        checkpoint_path : str, optional
            path of the checkpoint to load from or save to 
            (w.r.t current working dir.), by default 'segmentation.pth'
        metric : str, by default 'loss'
            metric to use for saving the best network, 
            'loss': min valid loss is used,
            'accuracy': max valid accuracy is used
        start_from_checkpoint : bool, optional
            start training form checkpoint, by default False
        lr_scheduler : torch lr scheduler, optional
            learning rate scheduler, by default None
        logging : dict, optional
            log training data to wandb. To enable logging provide the log_dict. 
            Logging will prompt you to provide login credentials for wandb, by default False

            Sample log_dict:
            log_dict={
                'name': 'Package',                  # Name of each run (change at each run)
                'entity': 'MML',          # username of wandb account
                'project': 'package_test',          # project name
                'notes': 'Test run',                # adding notes to the run
                'tags': ['Test'],                   # adding tags to runs for grouping
                                                    # list of tags can have multiple tags

                'log_histograms': False       # log network parameters and weights (True/False)
                }
        """
                
        fit_network(self, 
                    train_loader=train_loader, 
                    valid_loader=valid_loader, 
                    criterion=criterion,
                    acc_metric=acc_metric,
                    optimizer=optimizer, 
                    num_epochs=num_epochs, 
                    checkpoint_path=checkpoint_path, 
                    metric=metric,
                    start_from_checkpoint=start_from_checkpoint, 
                    lr_scheduler=lr_scheduler, 
                    log_dict=log_dict)

    def summary(self, img_dim):
        """
        Calling this method on a network gives the 
        network summary based on the input dimensions

        Parameters
        ----------
        img_dim : tuple
            (height, width)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch_summary(self.to(device), (self.img_ch, img_dim[0], img_dim[1])))


    def load(self, checkpoint_path):
        """
        Calling this method on network object will
        load the network from the checkpoint.

        Parameters
        ----------
        checkpoint_path : string
            checkpoint_path w.r.t. current directory
        """
        load_network(self, checkpoint_path)