import os
import sys
import torch
import wandb

from torchvision import transforms


def print_overwrite(step, total_step, loss, acc, operation):
    """
    Prints the running loss and accuracy for each step during training

    Parameters
    ----------
    step : int
        current step count
    total_step : int
        total step count
    loss : float
        running loss
    acc : float
        running accuracy
    operation : string
        'train' or 'valid'
    """
    sys.stdout.write('\r')

    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f  Accuracy: %.2f" %
                         (step, total_step, loss, acc))

    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f  Accuracy: %.2f" %
                         (step, total_step, loss, acc))

    sys.stdout.flush()


def save_checkpoint(epoch, network, optimizer, loss_valid, acc_valid, 
                    loss_min, acc_max, metric, ckp_path):
    """
    Saves the network checkpoint containing current epoch,
    network parameters, learning rate and current loss.
    Saves two checkpoints, one at every epoch 'checkpoint_path.pth'
    and other only when metric reaches an optimum value, 'checkpoint_path_best.pth'.

    Parameters
    ----------
    epoch : int
        current epoch
    network : nn.Module
        network to save the weights
    optimizer : torch.optim
        optimizer
    loss_valid : float
        validation loss
    acc_valid : float
        validation accuracy
    loss_min : float
        minimum validation loss
    acc_max : float
        maximum validation accuracy
    metric : string
        metric to use for saving the best network
    ckp_path : string 
        checkpoint path to save the weights to

    Returns
    -------
    loss_min : float
        minimum validation loss
    """

    state_dict = {'epoch': epoch,
                  'network': network.state_dict(),
                  'optimizer': optimizer,
                  'loss': loss_valid}

    if ckp_path:
        filename = os.path.join(os.getcwd(), ckp_path)
        torch.save(state_dict, filename)

        # save the network when valid loss in minimum
        if metric == 'loss':
            if loss_valid < loss_min:
                loss_min = loss_valid
                acc_max = max(acc_max, acc_valid)

                print("Min Loss: {:.4f} [Checkpoint Saved]".format(loss_min))
                if ckp_path.endswith('best.pth'):
                    filename = os.path.join(os.getcwd(), ckp_path)
                else:
                    filename = os.path.join(
                        os.getcwd(), ckp_path[:-4] + '_best.pth')
                torch.save(state_dict, filename)
        
        # save the network when valid accuracy in maximum
        elif metric == 'accuracy':
            if acc_valid > acc_max:
                acc_max = acc_valid
                loss_min = min(loss_min, loss_valid)
                
                print("Max Acc: {:.4f} [Checkpoint Saved]".format(acc_max))
                if ckp_path.endswith('best.pth'):
                    filename = os.path.join(os.getcwd(), ckp_path)
                else:
                    filename = os.path.join(
                        os.getcwd(), ckp_path[:-4] + '_best.pth')
                torch.save(state_dict, filename)
    else:
        loss_min = min(loss_min, loss_valid)
        acc_max = max(acc_max, acc_valid)
    return loss_min, acc_max


def load_network(network, checkpoint_path):
    """
    Loads the network from the checkpoint

    Parameters
    ----------
    checkpoint_path : string
        checkpoint_path w.r.t. current directory

    Returns
    -------
    None
    """
    checkpoint = torch.load(checkpoint_path)
    network.load_state_dict(checkpoint['network'])


def log_wandb(epoch, loss_train, loss_valid, loss_min, 
              acc_train, acc_valid, acc_max, lr_scheduler):
    """
    Logs training and validation data in wandb after epoch.

    Parameters
    ----------
    epoch : int
        current epoch
    loss_train : float
        training loss
    loss_valid : float
        validation loss
    loss_min : float
        minimum validation loss
    acc_max : float
        maximum validation accuracy
    lr_scheduler : torch scheduler
        to find the current learning rate during scheduling
    """
    log = {
            "Epoch": epoch,
            "Train Loss": loss_train,
            "Valid Loss": loss_valid,
            "Min Loss": loss_min,
            "Train Acc": acc_train,
            "Valid Acc": acc_valid,
            "Max Acc": acc_max,
        }

    if lr_scheduler:
        log['Learning Rate'] = lr_scheduler.state_dict()['_last_lr'][0]
    wandb.log(log)


def quantize_mask(mask):
    """
    Quantizes a given single or multi-channel
    mask from the network output.

    Parameters
    ----------
    mask : torch.tensor of shape(channel, height, width)
        output of network

    Returns
    -------
    mask: torch.tensor of shape(height, width)
        quantized mask
    """
    img_size = mask.shape
    if img_size[0] == 1:
        ones = torch.ones(img_size[1], img_size[2])
        zeros = torch.zeros(img_size[1], img_size[2])
        mask = torch.where(mask > 0.5, ones, zeros)
    else:
        mask = torch.argmax(mask, axis=0)
    return mask


class Denormalize(transforms.Normalize):
    """
    Undoes the normalization and returns 
    the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
