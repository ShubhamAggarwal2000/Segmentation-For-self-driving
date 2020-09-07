import torch
import torch.nn.functional as F


class IOU():
    """Calculates Intersection over Union
    between prediction and true_mask(non 1 hot encoded)

    Parameters
    ----------
    pred_mask : torch.tensor of shape [batch, channel, height, width]
        prediction from the network
    true_mask : torch.tensor of shape [batch, height, width]
        true mask from the dataset

    Returns
    -------
    float
        IOU between 0 and 100
    """
    def __init__(self):
        pass

    def __call__(self, pred_mask, true_mask):
        num_classes = pred_mask.shape[1]
        pred_mask = F.softmax(pred_mask, dim=1)
        true_mask = F.one_hot(true_mask.long(), num_classes).permute(0, 3, 1, 2)
        pred_mask = pred_mask.reshape(-1).detach()
        true_mask = true_mask.reshape(-1)
        intersection = (pred_mask * true_mask).sum()
        union = pred_mask.sum() + true_mask.sum() - intersection
        iou = (intersection + 1.0) / (union + 1.0)
        return iou * 100

class PixelWiseAccuracy():
    """
    Returns pixel wise accuracy between predicted
    and true mask for segmentation.

    Parameters
    ----------
    pred : torch.tensor of shape [batch, channel, height, width]
        predicted mask
    mask : torch.tensor of shape [batch, height, width]
        true mask

    Returns
    -------
    float between 0 and 100
        accuracy
    """
    def __init__(self):
        pass
    
    def __call__(self, pred, mask):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_size = pred.shape[0]
        num_channels = pred.shape[1]
        img_size = (pred.shape[2], pred.shape[3])

        # for binary class segmentation 
        if num_channels == 1:

            mask = mask.float()/torch.max(mask)
            ones = torch.ones(batch_size, 1, img_size[0], img_size[1]).to(device)
            zeros = torch.zeros(batch_size, 1, img_size[0], img_size[1]).to(device)
            pred = torch.where(pred > 0.5, ones, zeros).squeeze(1)
            correct = pred.eq(mask)
            accuracy = correct.to(torch.float32).mean().item() * 100

        # for multi class segmentation
        else:
            pred = torch.argmax(pred, axis=1)
            correct = pred.eq(mask)
            accuracy = correct.to(torch.float32).mean().item() * 100
        return accuracy
