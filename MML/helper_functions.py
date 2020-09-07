import os
import sys
import torch
import numpy as np
import wandb
from MML.utils.utils import print_overwrite, log_wandb, save_checkpoint
from MML.utils.decorators import timeit


def train(network, criterion, optimizer, train_loader, acc_metric):
    """
    Trains the network for 1 epoch

    Parameters
    ----------
    network : nn.Module 
        A network you want to train
    criterion : nn.Module
        loss function 
    optimizer : torch.optim
        Optimizer to learn the weights of the network
    train_loader : torch dataloader
        loads images of shape [batch, channel, height, width]
    acc_metric : function 
        determined the accuracy of the model

    Returns
    -------
    torch tensor of shape [] (scalar)
        loss_train, acc_train
    """

    loss_train = 0
    acc_train = 0
    running_loss = 0
    running_acc = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    network.train()

    for step, (images, masks) in enumerate(train_loader, start=1):

        # move the images and masks to GPU
        images = images.to(device)
        masks = masks.to(device)

        logits = network(images)

        # clear all the gradients before calculating them
        optimizer.zero_grad()

        # find the loss for the current step
        loss_train_step = criterion(logits, masks)
        acc_train_step = acc_metric(logits, masks)

        # calculate the gradients
        loss_train_step.backward()

        # update the parameters
        optimizer.step()

        # calculate running loss and accuracy
        loss_train += loss_train_step.item()
        acc_train += acc_train_step
        running_loss = loss_train/step
        running_accuracy = acc_train/step

        print_overwrite(step, len(train_loader), running_loss,
                        running_accuracy, 'train')

    loss_train /= len(train_loader)
    acc_train /= len(train_loader)
    return loss_train, acc_train


def validate(network, criterion, valid_loader, acc_metric):
    """
    validates the network for 1 epoch through the validation dataset

    Parameters
    ----------
    network : nn.Module 
        A network you want to train
    criterion : nn.Module
        loss function 
    valid_loader : torch dataloader
        loads images of shape [batch, channel, height, width]
    acc_metric : function 
        determined the accuracy of the model

    Returns
    -------
    torch tensor of shape [] (scalar)
        loss_valid, acc_valid
    """
    loss_valid = 0
    acc_valid = 0
    running_loss = 0
    running_acc = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    network.eval()

    # turn the gradients off for validation
    with torch.no_grad():

        for step, (images, masks) in enumerate(valid_loader, start=1):
            # move the images and masks to GPU
            images = images.to(device)
            masks = masks.to(device)

            logits = network(images)

            # find the loss and acc for the current step
            loss_valid_step = criterion(logits, masks)
            acc_valid_step = acc_metric(logits, masks)

            # calculate running loss and accuracy
            loss_valid += loss_valid_step.item()
            acc_valid += acc_valid_step
            running_loss = loss_valid/step
            running_accuracy = acc_valid/step

            print_overwrite(step, len(valid_loader),
                            running_loss, running_accuracy, 'valid')

    acc_valid /= len(valid_loader)
    loss_valid /= len(valid_loader)
    return loss_valid, acc_valid


@timeit
def fit_network(network, train_loader, valid_loader, criterion, acc_metric,
                optimizer, num_epochs=5, checkpoint_path=None, metric='loss',
                start_from_checkpoint=False, lr_scheduler=None, log_dict={}):
    """
    Trains and validates the given network for specified number of epochs. 
    Includes logging, saving checkpoints and lr_scheduling facilities.

    Parameters
    ----------
    network : nn.Module
        network you want to train for segmentation
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
    ## Sanity Check ##
    if start_from_checkpoint == True and checkpoint_path is None:
        raise AssertionError(
            'start_from_checkpoint must be False when checkpoint_path is None')

    if metric.lower() not in ['loss', 'accuracy']:
        raise AssertionError("metric must be in ['loss', 'accuracy']")

    if not start_from_checkpoint:
        start_epoch = 1
        loss_min = np.inf
        print('Starting Training from Epoch: {}\n'.format(start_epoch))
    else:
        if not os.path.exists(checkpoint_path):
            raise AssertionError(
                'Checkpoint Path does not exist. Make start_from_checkpoint=False.')

        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        network.load_state_dict(checkpoint['network'])
        optimizer = checkpoint['optimizer']
        loss_min = checkpoint['loss']

        if start_epoch > num_epochs:
            raise AssertionError(
                'Increase the number of num of epochs beyond {} to continue training'.format(start_epoch))
        print('Resuming Training from Epoch: {}\n'.format(start_epoch))

    # if log_dict is not None, log to wandb
    if log_dict:
        wandb.login()
        # configuration parameters
        config_dict = {'network': network.__class__.__name__,
                       'criterion': criterion.__class__.__name__,
                       'acc_metric': acc_metric.__class__.__name__,
                       'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
                       'num_epochs': num_epochs,
                       'optimizer': optimizer.__class__.__name__,
                       'patience': lr_scheduler.state_dict()['patience'] if lr_scheduler else None,
                       'decay_factor': lr_scheduler.state_dict()['factor'] if lr_scheduler else None,
                       'metric': metric,
                       'dataset': train_loader.dataset.__dict__['dataset'].name}

        # initialization parameters
        wandb.init(name=log_dict['name'],
                   entity=log_dict['entity'],
                   project=log_dict['project'],
                   notes=log_dict['notes'],
                   tags=log_dict['tags'],

                   config=config_dict)

        # if you want to log network weight histograms
        if log_dict['log_histograms']:
            print('\nLogging Network Parameters\n')
            wandb.watch(network)

    loss_min = np.inf
    acc_max = 0
    for epoch in range(start_epoch, num_epochs+1):

        loss_train, acc_train = train(network,
                                      criterion,
                                      optimizer,
                                      train_loader,
                                      acc_metric)

        loss_valid, acc_valid = validate(network,
                                         criterion,
                                         valid_loader,
                                         acc_metric)

        # Reduce the learning rate on Plateau
        if lr_scheduler:
            lr_scheduler.step(loss_valid)

        sys.stdout.write('\r')
        sys.stdout.flush()
        print('\n----------------------------------------------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Train Acc: {:.4f}  Valid Loss: {:.4f}  Valid Acc: {:.4f}'.format(
            epoch, loss_train, acc_train, loss_valid, acc_valid))
        print('----------------------------------------------------------------------------------------')

        # Save checkpoint
        loss_min, acc_max = save_checkpoint(epoch, network, optimizer, loss_valid, acc_valid,
                                            loss_min, acc_max, metric, checkpoint_path)

        # log the performance to wandb
        if log_dict:
            log_wandb(epoch, loss_train, loss_valid, loss_min,
                      acc_train, acc_valid, acc_max, lr_scheduler)
    print('\n---Training Complete---')
