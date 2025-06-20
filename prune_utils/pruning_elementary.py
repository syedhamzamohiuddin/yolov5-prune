import torch.nn.utils.prune as prune
import torch
from torch import nn

def prune_conv(cv1, prune_rate):
    # cv1: torch.nn.Conv2D
    # assumes no skip connection to cv1's output
    # prunes cv1 and removes relevant weights from cv2
    prune.ln_structured(cv1, name="weight", amount=prune_rate, n=2, dim=0)
    kept_mask = cv1.weight_mask[:,0,0,0]==1

    # following are the chosen filters for cv1
    tmp1 = cv1.weight[kept_mask].detach().clone()
    
    # call remove since we no longer need mask attribute
    prune.remove(cv1,"weight")
    
    new_ch = torch.sum(kept_mask).item()
    bias1 = cv1.bias is not None # this will always be False in ultralytics yolov5s, but I am not hardcoding
    
    cv1 = nn.Conv2d(cv1.in_channels, new_ch, cv1.kernel_size, 
                    cv1.stride, cv1.padding, cv1.dilation, 
                    cv1.groups, bias1, cv1.padding_mode)
    
    cv1.weight = nn.Parameter(tmp1)
    
    return cv1,  kept_mask


def adjust_conv(kept_mask, cv2):
    # cv2: torch.nn.Conv2D
    # kept_mask: boolen tensor, true values for indexes to be kept
    # Assumes cv1 -> cv2 OR cv1 -> BN -> cv2
    # assumes no skip connection
    # adjusts cv2 based on the mask resulting from pruning cv1
    # returns the updates cv2 layer along recieved mask for further use
    
    tmp2 = cv2.weight[:,kept_mask].detach().clone()
    new_ch = torch.sum(kept_mask).item()
    bias2 = cv2.bias is not None # this will always be False in ultralytics yolov5s, but I am not hardcoding
    cv2 = nn.Conv2d(new_ch, cv2.out_channels, cv2.kernel_size, cv2.stride,
                    cv2.padding, cv2.dilation, cv2.groups, bias2, cv2.padding_mode)
    cv2.weight = nn.Parameter(tmp2)

    return cv2, kept_mask

def adjust_conv_v2(kept_mask, cv2):
    # cv2: torch.nn.Conv2D
    # kept_mask: boolen tensor, true values for indexes to be kept
    # Assumes cv1 -> cv2 OR cv1 -> BN -> cv2
    # assumes no skip connection
    # adjusts cv2 based on the mask resulting from pruning cv1 of C3
    # returns the updates cv2 layer along recieved mask for further use
    
    tmp2 = cv2.weight[kept_mask].detach().clone()
    new_ch = torch.sum(kept_mask).item()
    bias2 = cv2.bias is not None # this will always be False in ultralytics yolov5s, but I am not hardcoding
    cv2 = nn.Conv2d(cv2.in_channels, new_ch, cv2.kernel_size, cv2.stride,
                    cv2.padding, cv2.dilation, cv2.groups, bias2, cv2.padding_mode)
    cv2.weight = nn.Parameter(tmp2)

    return cv2, kept_mask

def adjust_bn(kept_mask, bn):
    # bn: torch.nn.BatchNorm2D
    # kept_mask: boolen tensor, true values for indexes to be kept
    # bn follows cv1
    # assumes no skip connection
    # adjusts bn based on the mask resulting from pruning cv1
    # returns the updates bn layer along recieved mask for further use
    new_bn = nn.BatchNorm2d(torch.sum(kept_mask).item())
    new_bn.weight = nn.Parameter(bn.weight[kept_mask].clone())
    new_bn.bias = nn.Parameter(bn.bias[kept_mask].clone())
    new_bn.running_mean = bn.running_mean[kept_mask].clone()
    new_bn.running_var = bn.running_var[kept_mask].clone()

    return new_bn, kept_mask

def prune_conv_yolo(cv,prune_rate, kept_mask_prev=None):
    """
    cv : models.common.Conv of yolov5 repo => It has structure Conv2D -> BatchNorm2D -> activation
    Modifies Conv in-place and returns kept_mask
    Note: kept_mask must be used to adjust the following Conv2D layer
    
    """
    if kept_mask_prev is not None:
        adjust_conv_yolo(cv, kept_mask_prev)
    
    pruned_conv,  kept_mask = prune_conv(cv.conv, prune_rate)
    cv.conv = pruned_conv
    adjusted_bn, kept_mask = adjust_bn(kept_mask, cv.bn)
    cv.bn = adjusted_bn
    #print(cv.bn)

    return kept_mask

def adjust_conv_yolo(cv, kept_mask):
    """
    cv : models.common.Conv of yolov5 repo => It has structure Conv2D -> BatchNorm2D -> activation
    Adjusts Conv in-place. Only Conv2D of Conv needs to be adjusted because it directly follows the pruned Conv2D
    """
    pruned_conv,  kept_mask = adjust_conv(kept_mask, cv.conv)
    cv.conv = pruned_conv


def adjust_conv_yolo_v2(cv, kept_mask):
    """
    cv : models.common.Conv of yolov5 repo => It has structure Conv2D -> BatchNorm2D -> activation
    Adjusts Conv in-place. Only Conv2D of Conv needs to be adjusted because it directly follows the pruned Conv2D
    """
    pruned_conv,  kept_mask = adjust_conv_v2(kept_mask, cv.conv)
    cv.conv = pruned_conv
