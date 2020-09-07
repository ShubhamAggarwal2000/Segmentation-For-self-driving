import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """
    Binary Cross Entropy Loss for Segmentation 
    (Used for single class segmentation)

    Parameters
    ----------
    mask_pred : torch.tensor of shape [batch, channel, height, width]
        predicted mask from the network

    mask_true : torch.tensor of shape [batch, 1, height, width]
        true mask from the dataset

    Returns
    -------
    loss : torch tensor of shape []
    """

    def __init__(self):
        super(BCELoss, self).__init__()
        self.BCE = nn.BCELoss()

    def forward(self, mask_pred, mask_true):
        mask_pred = mask_pred.float().view(-1)
        mask_true = mask_true.float().view(-1)/torch.max(mask_true)
        loss = self.BCE(mask_pred, mask_true)
        return loss

class CELoss(nn.Module):
    """
    Cross Entropy Loss for Segmentation 
    (Used for multi-class segmentation)
    Make sure to not use softmax in the network
    as CELoss automatically softmaxes the network output.

    Parameters
    ----------
    pred : torch.tensor of shape [batch, channel, height, width]
        predicted mask from the network

    mask : torch.tensor of shape [batch, 1, height, width]
        true mask from the dataset

    Returns
    -------
    loss : torch tensor of shape []
    """

    def __init__(self):
        super(CELoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, mask):
        req_classes = pred.shape[1]
        pred = pred.permute(0, 2, 3, 1).reshape(-1, req_classes)
        mask = mask.reshape(-1).long()
        loss = self.cross_entropy(pred, mask)
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for Segmentation 
    (Used for multi-class segmentation)
    Make sure to not use softmax in the network
    as CELoss automatically softmaxes the network output.

    Parameters
    ----------
    pred : torch.tensor of shape [batch, channel, height, width]
        predicted mask from the network

    mask : torch.tensor of shape [batch, 1, height, width]
        true mask from the dataset

    Returns
    -------
    loss : torch tensor of shape []
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, pred, mask, smooth=1):
        

        inputs = F.sigmoid(pred)       
        inputs = inputs.view(-1)
        targets = mask.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice


class IOULoss():
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
    def _init_(self):
        pass

    def _call_(self, pred_mask, true_mask):
        num_classes = pred_mask.shape[1]
        pred_mask = F.softmax(pred_mask, dim=1)
        true_mask = F.one_hot(true_mask.long(), num_classes).permute(0, 3, 1, 2)
        pred_mask = pred_mask.reshape(-1)
        true_mask = true_mask.reshape(-1)
        intersection = (pred_mask * true_mask).sum()
        union = pred_mask.sum() + true_mask.sum() - intersection
        iou = (intersection + 1.0) / (union + 1.0)
        return 1 - iou

       
