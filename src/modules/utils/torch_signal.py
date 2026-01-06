import torch
import torch.nn.functional as F


def padding(x, frame_length, hop_length):
    """pading the end of x with zeros
    Shapes:
        x: [n_batch, n_sample]    
    Ref:
        https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/ops/signal/shape_ops.py#L166
    """    
    if not torch.is_tensor(x):
        raise TypeError("X must be an tensor, given {}".format(type(x)))
    
    _, n_sample = x.shape    
    n_frames = -(-n_sample // hop_length) # Using double negatives to round up.
    n_padding = frame_length + hop_length * (n_frames - 1) - n_sample
    x_padded = F.pad(x, (0, n_padding), mode='constant', value=0)
    return x_padded
