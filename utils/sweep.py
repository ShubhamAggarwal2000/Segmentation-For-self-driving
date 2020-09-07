import os
import sys
import torch
import numpy as np
import wandb

sys.path.append('.')
from MML.utils.decorators import timeit
from MML.helper_functions import train, validate

class SweepNetwork():
    """
    Class to run each section of hyperparameter sweep 
    (train a network for a given number of epoch)

    Parameters
    ----------
    network : List of networks

    optimizer : List of optimizer classes

    train_loader : dataloader object

    valid_loader : dataloader object

    criterion : list of criterion objects
        
    acc_metric : accuracy metric object (not a list)
        
    log_dict : log_dict : Logging dict
    """
    def __init__(self, network, optimizer, train_loader, 
                 valid_loader, criterion, acc_metric, log_dict):
        self.network = network
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.acc_metric = acc_metric
        self.log_dict = log_dict

        # dictionary to create mappings from name of classes to indices in the lists
        self.network_mapping = {x.__class__.__name__ : i for i, x in enumerate(network)}
        self.criterion_mapping = {x.__class__.__name__ : i for i, x in enumerate(criterion)}
        self.optimizer_mapping = {x.__name__ : i for i, x in enumerate(optimizer)}

    @timeit
    def __call__(self):

        # WandB – Initialize a new run
        wandb.init(name=self.log_dict['name'], 
                  entity=self.log_dict['entity'], 
                  notes=self.log_dict['notes'],
                  tags=self.log_dict['tags'])

        network = self.network[self.network_mapping[wandb.config.network]]

        criterion = self.criterion[self.criterion_mapping[wandb.config.criterion]]

        optimizer = self.optimizer[self.optimizer_mapping[wandb.config.optimizer]](network.parameters(), 
                                                           wandb.config.learning_rate)

        # If patience and decay_factor is not None, use lr_scheduler
        if wandb.config.patience and wandb.config.decay_factor:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            factor=wandb.config.decay_factor, 
                                                            patience=wandb.config.patience)
        else:
            lr_scheduler = None

        loss_min = np.inf
        acc_max = 0
        for epoch in range(1,wandb.config.num_epochs+1):
            
            loss_train, acc_train = train(network, criterion, optimizer, self.train_loader, self.acc_metric)
            loss_valid, acc_valid = validate(network, criterion, self.valid_loader, self.acc_metric)
            
            sys.stdout.write('\r')
            sys.stdout.flush()
            print('\n----------------------------------------------------------------------------------------')
            print('Epoch: {}  Train Loss: {:.4f}  Train Acc: {:.4f}  Valid Loss: {:.4f}  Valid Acc: {:.4f}'.format(
                epoch, loss_train, acc_train, loss_valid, acc_valid))
            print('----------------------------------------------------------------------------------------')
            
            if wandb.config.patience and wandb.config.decay_factor:
                lr_scheduler.step(loss_valid)

            # log data to wandb
            loss_min, acc_max = log_wandb(epoch, loss_train, loss_valid, acc_train, acc_valid, 
                      loss_min, acc_max, lr_scheduler, wandb.config.learning_rate)

        # delete the network and optimizer after each 
        # section of sweep to free up some space
        del network
        del optimizer


def generate_sweep_config(network, num_epochs, criterion, acc_metric, optimizer, learning_rate, 
                          train_loader, patience=None, decay_factor=None, metric=None):
    """
    Generates sweep configuration dictionary.

    Parameters
    ----------
    network : List of networks

    num_epochs : List of ints
    
    criterion : List of criterion objects

    acc_metric : accuracy metric object (not a list)

    optimizer : List of optimizer classes

    train_loader : dataloader object

    learning_rate : List of learning rates

    patience = [3] : patience for learning rate scheduler
    decay_factor = [0.8] : decay factor for learning rate scheduler

    metric : The metric to be optimized, optional, by default None
            values must be in ["loss", accuracy", None]

    Returns
    -------
    dict
        sweep_config
    """

    if metric.lower() not in ["loss", "accuracy" , None]:
        raise AssertionError("Metric must be in ['loss', 'accuracy' , None]")

    if metric.lower() == 'loss':
        metric = 'Min Loss'
        goal = 'minimize'

    elif metric.lower() == 'accuracy':
        metric = 'Max Acc'
        goal = 'maximize'

    else:
        metric = None
        goal = None

    sweep_config = {
        'method': 'grid',

        'metric': {
        'name': metric,
        'goal': goal,   
        },

        'parameters': {
            'metric': {
                'values': [metric]
            },
            'learning_rate': {
                'values': learning_rate
            },
            'num_epochs': {
                'values': num_epochs
            },
            'network': {
                'values': [x.__class__.__name__ for x in network]
            },
            'criterion': {
                'values': [x.__class__.__name__ for x in criterion]
            },
            'acc_metric': {
                'values': [acc_metric.__class__.__name__]
            },
            'optimizer': {
                'values': [x.__name__ for x in optimizer]
            },
            'patience': {
                'values': patience if patience else [None]
            },
            'decay_factor': {
                'values': decay_factor if decay_factor else [None]
            },
            'dataset': {
                'values': [train_loader.dataset.__dict__['dataset'].name]
            }
        }
    }
    return sweep_config


def log_wandb(epoch, loss_train, loss_valid, acc_train, acc_valid, 
              loss_min, acc_max, lr_scheduler, learning_rate):
    
    loss_min = min(loss_min, acc_valid)
    acc_max = max(acc_max, acc_valid)

    log = {
        "Epoch": epoch,
        "Train Loss": loss_train,
        "Valid Loss": loss_valid,
        "Min Loss": loss_min, 
        "Train Acc": acc_train,
        "Valid Acc": acc_valid,
        "Max Acc": acc_max, 
    }

    # if learning rate scheduler is available,
    # get the learning rate from that
    # else learning rate remains constant
    if lr_scheduler:
        log["Learning Rate"]=lr_scheduler.state_dict()['_last_lr'][0]
    else:
        log["Learning Rate"]=learning_rate

    wandb.log(log)
    return loss_min, acc_max

@timeit
def sweep(network, num_epochs, criterion, acc_metric, optimizer, train_loader, valid_loader, 
          learning_rate, log_dict, patience=None, decay_factor=None, metric=None):
    """Function for hyperparameter sweep

    Parameters
    ----------
    network : List of networks

    num_epochs : List of ints
    
    criterion : List of criterion

    acc_metric : accuracy metric object

    optimizer : List of optimizers

    train_loader : dataloader object

    valid_loader : dataloader object

    learning_rate : List of learning rates

    patience = [3] : patience for learning rate scheduler
    decay_factor = [0.8] : decay factor for learning rate scheduler

    metric : The metric to be optimized, optional, by default None
            values must be in ["loss", accuracy", None]

    log_dict : Logging dict

    Sample configuration to be fed into the sweep function

        network = [DRUNet(3, 5), AttentionUNet(3, 5)]
        criterion = [CELoss()]
        optimizer = [torch.optim.Adam]
        num_epochs = [20]
        learning_rate = [0.001]
        patience = [3]
        decay_factor = [0.8]

        log_dict = {
            'name': 'drunet_sweep',
            'entity': 'MML',
            'project': 'sweep'
            'notes': 'notes for run'
            'tags': ['tag'],
        }
    """
    
    # generate config_dict
    sweep_config = generate_sweep_config(network=network,
                                        num_epochs=num_epochs,
                                        criterion=criterion,
                                        acc_metric = acc_metric,
                                        optimizer=optimizer,
                                        patience=patience,
                                        decay_factor=decay_factor,
                                        learning_rate=learning_rate,
                                        metric=metric,
                                        train_loader=train_loader)
    
    # Instantiate the SweepNetwork class to run the sweep
    sweep_network = SweepNetwork(network = network,
                    optimizer = optimizer,
                    train_loader = train_loader,
                    valid_loader = valid_loader,
                    criterion = criterion,
                    acc_metric = acc_metric,
                    log_dict = log_dict)
    
    # Run the sweep
    wandb.agent(wandb.sweep(sweep_config, project=log_dict['project']), function=sweep_network)