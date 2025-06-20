# Pruning the 'm' component of C3. m is sequential module containing one or more Bottlenecks
import torch.nn.utils.prune as prune
import torch
from torch import nn
from .pruning_elementary import prune_conv, adjust_conv, adjust_bn, prune_conv_yolo, adjust_conv_yolo, adjust_conv_yolo_v2

def prune_m_backbone(m, prune_rate):
    """
    Prunes submodule m
    Only prunes cv1 component component 'm' of C3
    m : torch.nn.Sequntial module containing Bottlenecks
    """
    for bottleneck in m:
        cv1 = bottleneck.cv1
        cv2 = bottleneck.cv2
        # prune and adjust Conv's components
        kept_mask = prune_conv_yolo(cv1, prune_rate)
        adjust_conv_yolo(cv2, kept_mask)
    

def prune_c3_backbone(c3, prune_rate=0.3, kept_mask_prev=None):
    """
    Prunes C3
    next_Conv: next module's first conv layer
    returns kept_mask of cv3, the last component of C3
    """
    cv1, cv2, cv3, m = c3.cv1, c3.cv2, c3.cv3, c3.m

    # Adjust the cv1 and cv2 of C3, if previous layer was pruned
    if kept_mask_prev is not None:
        adjust_conv_yolo(cv1, kept_mask_prev)
        adjust_conv_yolo(cv2, kept_mask_prev)

    
    # prune cv1
    kept_mask_cv1 = prune_conv_yolo(cv1, prune_rate)
    
    # prune cv2
    kept_mask_cv2 = prune_conv_yolo(cv2, prune_rate)

    # prune bottlenecks
    prune_m_backbone(m, prune_rate) # prunes inplace. only prunes cv1

    # prune cv3
    kept_mask_cv3 = prune_conv_yolo(cv3, prune_rate)

    # adjust m => adjust all bottlenecks. pruning cv1 of C3 requires adjusting all bottlenecks
    # Bottlenecks require special handling/adjusting
    ## Bottlenecks, using the same prev_mask, 
    ### cv1 require adjusting channels_in as usual, however, cv2 requires adjusting channels_out based on same mask
    #### You have to understand that that if cv1 of C3 is pruned, then using the same mask, all cv2s of all Bottlenecks must be pruned
    #### So last bottleneck's cv2's out channels are also this adjusted by similar mask
 
    for b in m:

        # Bottleneck's component
        cv1_b = b.cv1
        cv2_b = b.cv2
        
        adjust_conv_yolo(cv1_b, kept_mask_cv1)
        adjust_conv_yolo_v2(cv2_b, kept_mask_cv1)
        
        # Since out_channels of bottleneck's cv2 are affected, batchnorm must also be fixed
        adjusted_bn,_ = adjust_bn(kept_mask_cv1, cv2_b.bn)
        cv2_b.bn = adjusted_bn
    # adjust cv3
    adjust_conv_yolo(cv3, torch.cat((kept_mask_cv1, kept_mask_cv2)))

    return kept_mask_cv3

def prune_c3_head(c3, prune_rate=0.3, kept_mask_prev=None):
    """
    Prunes C3
    next_Conv: next module's first conv layer
    returns kept_mask of cv3, the last component of C3
    """
    cv1, cv2, cv3, m = c3.cv1, c3.cv2, c3.cv3, c3.m

    # Adjust the cv1 and cv2 of C3, if previous layer was pruned
    if kept_mask_prev is not None:
        adjust_conv_yolo(cv1, kept_mask_prev)
        adjust_conv_yolo(cv2, kept_mask_prev)

    
    # prune cv1
    kept_mask_cv1 = prune_conv_yolo(cv1, prune_rate)
    
    # prune cv2
    kept_mask_cv2 = prune_conv_yolo(cv2, prune_rate)

    # prune bottlenecks
    kept_mask_m=prune_m_head(m, prune_rate) # prunes inplace. only prunes cv1

    # prune cv3
    kept_mask_cv3 = prune_conv_yolo(cv3, prune_rate)

    # adjust the first bottleneck in m
    adjust_conv_yolo(m[0].cv1, kept_mask_cv1)
        
    # adjust cv3
    adjust_conv_yolo(cv3, torch.cat((kept_mask_m, kept_mask_cv2))) # This order of concatenation in C3's forward method, so same order of filtering order is achieved through this
    

    return kept_mask_cv3


def prune_sppf_backbone(sppf, prune_rate=0.3, kept_mask_prev=None):
    """
    forward method of SPPF
    x = cv1(x)
    y1 = m(x) # m is maxpool
    y2 = m(y1)
    self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
    
    """
    # adjust conv1 of sppf, the first layer here, that's connected to the previous component that is pruned
    if kept_mask_prev is not None:
        adjust_conv_yolo(sppf.cv1, kept_mask_prev)

    # Prune cv1
    kept_mask_cv1 = prune_conv_yolo(sppf.cv1, prune_rate)

    # Prune cv2
    kept_mask_cv2 = prune_conv_yolo(sppf.cv2, prune_rate)

    # Adjust cv2
    adjust_conv_yolo(sppf.cv2, torch.cat((kept_mask_cv1,kept_mask_cv1,kept_mask_cv1,kept_mask_cv1)))

    return kept_mask_cv2


def prune_m_head(m, prune_rate):
    """
    Prunes submodule m
    Only prunes component 'm' of C3
    m : torch.nn.Sequntial module containing Bottlenecks
    """
   
    #prune cv1 and adjust cv2
    for bottleneck in m:
        cv1 = bottleneck.cv1
        cv2 = bottleneck.cv2
        # prune and adjust Conv's components
        kept_mask = prune_conv_yolo(cv1, prune_rate)
        adjust_conv_yolo(cv2, kept_mask)

    # Prune cv2 of this bottleneck and adjust cv1 of the next bottleneck
    for b1, b2 in zip(m[:-1], m[1:]):
        conv1 = b1.cv2
        conv2 = b2.cv1

        # prune and adjust Conv's components
        kept_mask = prune_conv_yolo(conv1, prune_rate)
        adjust_conv_yolo(conv2, kept_mask)

    # Prune cv2 of last bottleneck and adjust cv3 of C3
    ## Prune cv2 if bottleneck
    b_last = m[-1].cv2
    kept_mask = prune_conv_yolo(b_last, prune_rate)
     
    return kept_mask
