import random
import numpy as np
import wandb
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from MML.utils.decorators import timeit
from MML.utils.utils import quantize_mask, Denormalize


def visualize_dataset(dataset, index=None, alpha=0.1, transform=None, colorbar=False):
    """
    Used for visualizing the dataset

    Parameters
    ----------
    dataset : torch dataset
        the dataset to be visualizaed
    index : int, optional
        index of any specific data that needs to be visualized, 
        If not specified it displays random data from the dataset, by default None
    alpha : float, optional
        opaqueness value, higher the value more opaque will be 
        the mask imposition on image, by default 0.1
    transform : transform object, optional
        used to denormalize the image for visualization. 
        If not passed, normalized image will be displayed
    colorbar : whether we want to display colorbar, 
        by default True
    """

    # get the image and true mask from the dataset
    image, mask = dataset[index if index else random.randint(0, len(dataset))]

    # if transform is not None, denormalize the image
    if transform:
        mean = transform.transform_dict['mean']
        std = transform.transform_dict['std']
        denormalize_image = Denormalize(mean, std)
        image = denormalize_image(image)

    # if the image has shape (1, height, width),
    # reshape it to (height, width) and convert to numpy
    if image.numpy().shape[0] == 1:
        image = image.numpy().squeeze(0)

    # if the image has shape (3, height, width),
    # reshape it to (height, width, 3) and convert to numpy
    else:
        image = image.numpy().transpose(1, 2, 0)

    # convert mask to numpy
    mask = mask.numpy()

    # if the mask is binary, normalize it to [0, 1]
    if len(np.unique(mask)) == 2:
        mask = mask/np.max(mask)

    # generate discrete cmap
    cmap = plt.get_cmap('Paired', np.max(mask.astype('uint8'))-np.min(mask.astype('uint8'))+1)

    plt.figure(figsize=(30, 8))

    # plot the image
    plt.subplot(1, 3, 1)
    plt.imshow(image, 'gray')
    plt.title('Train Image', fontdict={'fontsize': 32})
    plt.axis('off')

    # plot the mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap=cmap, vmin = np.min(mask)-.5, vmax = np.max(mask)+.5)
    if colorbar:
        plt.colorbar(ticks=np.arange(np.min(mask), np.max(mask)+1), 
                                orientation='vertical', fraction=0.05)
    plt.title('True Mask', fontdict={'fontsize': 32})
    plt.axis('off')

    # plot the overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image, 'gray')
    plt.imshow(mask, cmap=cmap, alpha=alpha, vmin = np.min(mask)-.5, vmax = np.max(mask)+.5)
    plt.title('Overlay', fontdict={'fontsize': 32})
    plt.axis('off')



@timeit
def visualize_prediction(network, dataset, index=None, alpha=0.1, 
                        quantize=True, mean=None, std=None, colorbar=False):
    """
    Used for visualizing the network predicted mask

    Parameters
    ----------
    network : nn.Module
        network whose prediction to visualize
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
    transform : transform object, optional
        used to denormalize the image for visualization. 
        If not passed, normalized image will be displayed
    colorbar : whether we want to display colorbar, 
        by default True
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # try to get just image from the dataset (test_dataset)
    try:
        image = dataset[index if index else random.randint(0, len(dataset))]
    
    # if that doesn't work, get image and mask in a list and just use the image (train_dataset)
    except:
        image = list(dataset[index if index else random.randint(0, len(dataset))])[0]

    # pass the image through the network
    network = network.to(device)
    image = image.to(device).unsqueeze(0)
    mask = network(image)  # (1, channel, height, width)

    # if the predicted mask has multiple channels, 
    # softmax it to get the probability distribution 
    mask = F.softmax(mask, dim=1) if mask.shape[1] != 1 else mask

    image = image.cpu().detach().squeeze(0) # (channel, height, width)

    # if mean and std is available, denormalize the image
    if mean and std:
        denormalize_image = Denormalize(mean, std)
        image = denormalize_image(image)

    # if the image has shape (1, height, width),
    # reshape it to (height, width) and convert to numpy
    if image.numpy().shape[0] == 1:
        image = image.numpy().squeeze(0)  # (height, width)

    # if the image has shape (3, height, width),
    # reshape it to (height, width, 3) and convert to numpy
    else:
        image = image.numpy().transpose(1, 2, 0)  # (height, width, 3)

    # if quantize is False and mask has multiple channels
    # print each individual mask channel as a heatmap
    if not quantize and mask.shape[1] != 1:

        num_channels = mask.shape[1]
        mask = mask.cpu().detach().squeeze(0)  # (channel, height, width)
        
        plt.figure(figsize=(20, num_channels * 10))

        for channel in range(num_channels):

            plt.subplot(num_channels, 3, channel*3 + 1)
            plt.imshow(image, 'gray')
            plt.title('Test Image', fontdict={'fontsize': 22})
            plt.axis('off')

            plt.subplot(num_channels, 3, channel*3 + 2)
            plt.imshow(mask[channel,:,:])
            if colorbar:
                plt.colorbar(orientation='vertical', fraction=0.05)
            plt.title('Predicted Mask: Channel: {}'.format(channel), fontdict={'fontsize': 22})

            plt.axis('off')

            plt.subplot(num_channels, 3, channel*3 + 3)
            plt.imshow(image, 'gray')
            plt.imshow(mask[channel,:,:], alpha=alpha)
            plt.title('Prediction Overlay', fontdict={'fontsize': 22})
            plt.axis('off')

        # Break from the function  
        return None 

    # if quantize is False and mask has single channel
    # print single mask channel as a heatmap
    if not quantize and mask.shape[1] == 1:

        mask = mask.cpu().detach().squeeze(0).squeeze(0)  # (height, width)

        plt.figure(figsize=(20, 10))

        plt.subplot(1, 3, 1)
        plt.imshow(image, 'gray')
        plt.title('Test Image', fontdict={'fontsize': 22})
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask)
        if colorbar:
            plt.colorbar(orientation='vertical', fraction=0.05)
        plt.title('Predicted Mask', fontdict={'fontsize': 22})
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(image, 'gray')
        plt.imshow(mask, alpha=alpha)
        plt.title('Prediction Overlay', fontdict={'fontsize': 22})
        plt.axis('off')

        # Break from the function    
        return None 

    # if quantize is True, quantize the mask
    if quantize:
        mask = quantize_mask(mask.cpu().detach().squeeze(0))  # (height, width)

    mask = np.array(mask)

    # get discrete colormap
    cmap = plt.get_cmap('Paired', np.max(mask.astype('uint8'))-np.min(mask.astype('uint8'))+1)

    # plot the quantized mask
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(image, 'gray')
    plt.title('Test Image', fontdict={'fontsize': 22})
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap=cmap, vmin = np.min(mask)-.5, vmax = np.max(mask)+.5)
    if colorbar:
        plt.colorbar(ticks=np.arange(np.min(mask), np.max(mask)+1), 
                                orientation='vertical', fraction=0.05)
    plt.title('Predicted Mask', fontdict={'fontsize': 22})
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image, 'gray')
    plt.imshow(mask, cmap=cmap, alpha=alpha, vmin = np.min(mask)-.5, vmax = np.max(mask)+.5)
    plt.title('Prediction Overlay', fontdict={'fontsize': 22})
    plt.axis('off')

def visualize_wandb(network, dataset, split=1, mean=None, std=None, title="Head Segmentation"):
    """Function to visualize the masks in wandb

    Parameters
    ----------
    network : nn.Module
        network whose prediction to visualize
    dataset : torch dataset
        the dataset to be visualized
    split : int
        % of test images to be displayed in wandb
    """
    labels = ['Background', 'Skull', 'Midline Falx', 'CSP', 'Chorroid', 'Parenchyma']
    class_labels = {index: label for index, label in enumerate(labels)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)

    for i in range(int(len(dataset)*split)):
        image , mask_true = dataset[i] #image: (1,h,w) #mask_true: (h,w)
        image = image.to(device).unsqueeze(0) #image: (1,1,h,w)
        mask_pred = network(image) #mask_pred: (1,1,h,w)
        mask_pred = quantize_mask(mask_pred.cpu().detach().squeeze(0)) #mask_pred: (h,w)

        image = image.cpu().detach().squeeze(0) #image: (channel, height, width)
        if mean and std:
            denormalize_image = Denormalize(mean, std)
            image = denormalize_image(image)

        image = image.permute(1,2,0).numpy().squeeze(-1)

        wandb.log({title: wandb.Image(image, 
                masks={
                        "predictions" : {
                            "mask_data" : mask_pred.numpy(),
                            "class_labels" : class_labels},
                        "ground_truth" : {
                            "mask_data" : mask_true.numpy(),
                            "class_labels" : class_labels}
                        })})
