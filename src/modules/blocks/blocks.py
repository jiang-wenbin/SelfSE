import numpy as np
import math
import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn.modules.activation import MultiheadAttention
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from ..utils.common import EPS


class ModelBase(torch.nn.Module):
    def __init__(self, model_conf):
        super().__init__()
        self.model_conf = model_conf
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError


def apply_mask_bound(mask, mask_bound=None):
    # apply mask bounding
    if mask_bound in [None, 'none', 'None', 'null']:
        return mask
    if mask.is_complex():
        if mask_bound == 'mag_tanh':
            mag = mask.abs().clip(EPS)
            bounded_mag = torch.tanh(mag)  # tanh bounded
            phase = mask/mag
            mask = bounded_mag * phase
        elif mask_bound == 'complex_tanh':
            real_mask = torch.tanh(mask.real)
            imag_mask = torch.tanh(mask.imag)
            mask = torch.complex(real_mask, imag_mask)
        else:
            raise ValueError(f'Mask bound error {mask_bound}')
    else:
        if mask_bound == 'sigmoid':
            mask = torch.sigmoid(mask)
        elif mask_bound == 'tanh':
            mask = torch.tanh(mask)
        elif mask_bound == 'relu':
            mask = torch.relu(mask)
        else:
            raise ValueError(f'Mask bound error {mask_bound}')
    return mask


def padded_cat(x, y, dim=1):
    # Pad x to have same size with y, and cat them
    x_pad = F.pad(x, (0, y.shape[3] - x.shape[3], 0, y.shape[2] - x.shape[2]))
    z = torch.cat((x_pad, y), dim=dim)
    return z


def get_encoded_freqs(n_freqs, param):
    # param: [(in_channels, out_channels, kernel_size, stride), ...]
    for _, _, _, enc_stride in param:
        n_freqs = (n_freqs - 3) // enc_stride[0]
    return n_freqs


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features=257, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requires_grad = True

    def forward(self, x):
        # x: shape of [B, C, F, T]
        x = torch.einsum('bcft,f->bcft', [x, self.slope])
        return self.beta * torch.sigmoid(x)


class MaskBound(nn.Module):
    # module version of apply_mask_bound
    def __init__(self, mask_bound):
        super().__init__()
        self.mask_bound = mask_bound

    def forward(self, mask):
        return apply_mask_bound(mask, self.mask_bound)


class LayerNorm_permute(nn.Module):
    '''LayerNorm by permute'''

    def __init__(self, normalized_shape, permute, eps=1e-05):
        super().__init__()
        self.permute = permute
        self.layer_norm = nn.LayerNorm(normalized_shape, eps)

    def forward(self, x):
        x = x.permute(self.permute)
        x = self.layer_norm(x)
        x = x.permute(self.permute)
        return x


def get_norm(layer, n_channels, n_freqs, norm):
    if norm == 'weight':
        layer = weight_norm(layer)
        norm = nn.Identity()  # just a placeholder, do nothing
    elif norm == None:
        norm = nn.Identity()  # just a placeholder, do nothing
    elif norm == 'batch':
        norm = nn.BatchNorm2d(n_channels)
    elif norm == 'instance':
        norm = nn.InstanceNorm2d(n_channels)
    elif norm == 'layer':
        assert n_freqs > 1
        # layer norm in penult dim
        norm = LayerNorm_permute(n_freqs, (0, 1, 3, 2))
    else:
        raise ValueError(f'norm error {norm}')
    return layer, norm


class ConvBlock(nn.Module):
    '''norm: weight, batch, layer, instance'''

    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(2, 1),
                 padding=(0, 1), causal=True, norm='weight', n_freqs=0, 
                 act_fn='PReLU', time_emb_dim=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.conv, self.norm = get_norm(self.conv, out_channels, n_freqs, norm)
        self.activation = getattr(nn, act_fn)()
        self.causal = causal
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.time_emb_mlp = torch.nn.Sequential(self.activation, torch.nn.Linear(
            time_emb_dim, out_channels)) if time_emb_dim else None

    def forward(self, x, time_emb=None):
        '''
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        '''
        x = self.conv(x)
        if self.time_emb_mlp is not None and time_emb is not None:
            x += self.time_emb_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        if self.causal is True:
            x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class TransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(2, 1),
                 padding=(0, 0), output_padding=(0, 0), is_last=False, causal=True,
                 norm='weight', n_freqs=0, act_fn='PReLU', time_emb_dim=None):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                       padding, output_padding)
        self.conv, self.norm = get_norm(self.conv, out_channels, n_freqs, norm)
        self.is_last = is_last
        self.causal = causal
        self.activation = getattr(nn, act_fn)()
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.time_emb_mlp = torch.nn.Sequential(self.activation, torch.nn.Linear(
            time_emb_dim, out_channels)) if time_emb_dim else None

    def forward(self, x, time_emb=None):
        '''
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        '''
        x = self.conv(x)
        if self.time_emb_mlp is not None and time_emb is not None:
            x += self.time_emb_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        if self.causal is True:
            x = x[:, :, :, :-1]  # chomp size
        if self.is_last is False:
            x = self.norm(x)
            x = self.activation(x)
        return x


class SubPixelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=(3, 5), reshape=True, kernel_size=3, stride=1, padding=1):
        '''Upsampling using sub pixel layer
        input: shape of [n_batch, n_channel, n_freq, n_frame]
        output: shape of [n_batch, n_channel, ratio[0], ratio[1], n_freq, n_frame]
        '''
        super(SubPixelConv2d, self).__init__()
        self.out_channels = out_channels
        self.ratio = ratio
        self.reshape = reshape
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels*np.prod(ratio),
                                            kernel_size, stride, padding),
                                  nn.Tanh())

    def forward(self, x):
        out = self.conv(x)
        if self.reshape:
            n_batch, n_channels, n_freq, n_frame = out.shape
            out = out.view((n_batch, math.floor(n_channels/np.prod(self.ratio)),
                            self.ratio[0], self.ratio[1], n_freq, n_frame))
            out = out.contiguous()
        return out


class LinearUpSample(nn.Module):
    def __init__(self, n_freq, ratio=5):
        super().__init__()
        '''Upsampling using linear layer'''
        self.n_freq = n_freq
        self.ratio = ratio
        self.W = nn.Parameter(torch.ones(n_freq, n_freq*ratio))
        for p in self.parameters():
            nn.init.xavier_normal_(p)

    def forward(self, x):
        '''
        x: shape of [n_batch, n_channel, n_freq, n_frame]
        return: shape of [n_batch, n_channel, ratio, 1, n_freq, n_frame]
        '''
        n_batch, n_channel, n_freq, n_frame = x.shape
        # upsampling frequency
        out = torch.einsum('bcft,fg->bcgt', [x, self.W])
        out = out.view(n_batch, n_channel, self.ratio, n_freq, n_frame)
        return out.contiguous().unsqueeze(3)


class LSTM_Block(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, **kwargs):
        super().__init__()
        self.lstm_layer = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=batch_first, **kwargs)

    def forward(self, x):
        '''Input shape: [batch, ..., time]'''
        self.lstm_layer.flatten_parameters()
        batch_size, n_channels, n_freq, n_frames = x.shape
        lstm_in = x.permute(0, 3, 1, 2).reshape(
            batch_size, n_frames, n_channels*n_freq)
        lstm_out, _ = self.lstm_layer(lstm_in)
        lstm_out = lstm_out.reshape(
            batch_size, n_frames, n_channels, n_freq).permute(0, 2, 3, 1)
        return lstm_out

class DenseBlock(nn.Module):
    def __init__(self, in_channels=64, n_freqs=256, kernel_size=(3, 2), depth=4, **kwargs):
        '''
        input: shape of [n_batch, n_channels, n_freqs, n_frames]
        output: shape of [n_batch, n_channels, n_freqs, n_frames]
        '''
        super(DenseBlock, self).__init__()
        self.debug = kwargs.get('debug', False)
        self.dense_block = nn.ModuleList()
        t_step = kernel_size[-1]
        for i in range(depth):
            dilate = 2 ** i
            n_pad = t_step + (dilate - 1) * (t_step - 1) - 1
            if self.debug is True:
                print(f'layer_{i}: dilate {dilate} n_pad {n_pad}')
            self.dense_block.append(nn.Sequential(
                nn.ZeroPad2d((n_pad//2, n_pad - n_pad//2, 1, 1)),
                nn.Conv2d(in_channels*(i+1), in_channels,
                          kernel_size=kernel_size, dilation=(1, dilate)),
                nn.InstanceNorm2d(in_channels) if kwargs.get('norm', 'instance') else
                # layer norm in n_freqs dim
                LayerNorm_permute(n_freqs, (0, 1, 3, 2)),
                nn.PReLU(in_channels)))

    def forward(self, x):
        skip = x
        for i, layer in enumerate(self.dense_block):
            out = layer(skip)
            skip = torch.cat([out, skip], dim=1)
            if self.debug:
                print(f'layer_{i}:', out.shape, skip.shape)
        return out


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2):
        '''Upsampling using sub pixel layers, along with the frequency dimension
        input: shape of [n_batch, n_channels, n_freqs, n_frames]
        output: shape of [n_batch, n_channels, n_freqs*ratio, n_frames]
        '''
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.ratio = ratio
        self.conv = nn.Conv2d(in_channels, out_channels *
                              ratio, kernel_size=kernel_size)

    def forward(self, x):
        out = self.conv(x)
        n_batch, n_channels, n_freqs, n_frames = out.shape
        out = out.view((n_batch, n_channels//self.ratio,
                       self.ratio, n_freqs, n_frames))
        out = out.contiguous().view((n_batch, n_channels//self.ratio, -1, n_frames))
        return out


class SubPixelTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs=256, kernel_size=1, ratio=2, **kwargs):
        super().__init__()
        self.conv = SPConvTranspose2d(
            in_channels, out_channels, kernel_size, ratio)
        if kwargs.get('norm', 'instance'):
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            # layer norm in n_freqs dim
            self.norm = LayerNorm_permute(n_freqs, (0, 1, 3, 2))
        self.activation = nn.PReLU()
        self.is_last = kwargs.get('is_last', False)

    def forward(self, x):
        '''
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        '''
        x = self.conv(x)
        if self.is_last is False:
            x = self.norm(x)
            x = self.activation(x)
        return x


class DPBase(nn.Module):
    '''
    Deep dual-path base module
    '''

    def __init__(self, **kwargs):
        super().__init__()
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])

    def forward(self, input, time_emb=None):
        # input shape: batch, N, dim1, dim2
        # apply rnn on dim1 first and then dim2

        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = output.permute(0, 3, 2, 1).contiguous().view(
                batch_size*dim2, dim1, -1)  # B*dim2, dim1, N
            if time_emb is not None:
                row_output = self.row_rnn[i](row_input, timestep=time_emb)  # B*dim2, dim1, H
            else:
                row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = row_output.view(
                batch_size, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output

            col_input = output.permute(0, 2, 3, 1).contiguous().view(
                batch_size*dim1, dim2, -1)  # B*dim1, dim2, N
            if time_emb is not None:
                col_output = self.col_rnn[i](col_input, timestep=time_emb)  # B*dim1, dim2, H
            else:
                col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            col_output = col_output.view(
                batch_size, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()  # B, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output

        return output


class DPTransformer1(DPBase):
    '''
    Dual-path Transformer
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.debug = kwargs.pop('debug', False)
        num_layers = kwargs.pop('num_layers', 4)
        encoder_dim = kwargs.pop('encoder_dim', 64)
        kwargs.update({
            'd_model': encoder_dim,
            'nhead': kwargs.pop('nhead', 4),
            'dim_feedforward': kwargs.pop('dim_feedforward', 512),
            'batch_first': True
        })
        for _ in range(num_layers):
            self.row_rnn.append(TransformerEncoderLayer(**kwargs))
            self.col_rnn.append(TransformerEncoderLayer(**kwargs))
            self.row_norm.append(nn.GroupNorm(1, encoder_dim, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, encoder_dim, eps=1e-8))


class ImprovedTransformedLayer(nn.Module):
    '''
    Improved Transformer module as used in [1].
    It is Multi-Head self-attention followed by LSTM, activation and linear projection layer.

    Args:
        d_model (int): Number of input channels.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Number of neurons in the RNNs cell state.
            Defaults to 256. RNN here replaces standard FF linear layer in plain Transformer.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        activation (str, optional): activation function applied at the output of RNN.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).

    References
        [1] Chen, Jingjing, Qirong Mao, and Dong Liu. 'Dual-Path Transformer
        Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation.'
        arXiv (2020).
    '''

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation='relu',
                 bidirectional=True, batch_first=True):
        super(ImprovedTransformedLayer, self).__init__()

        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(d_model, dim_feedforward,
                          bidirectional=bidirectional, batch_first=batch_first)
        ff_inner_dim = 2 * dim_feedforward if bidirectional else dim_feedforward
        self.linear = nn.Linear(ff_inner_dim, d_model)
        self.activation = getattr(torch.nn.functional, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src: shape of [n_batch, n_seq, feature]
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # rnn is applied
        self.rnn.flatten_parameters()
        src2 = self.linear(self.dropout(self.activation(self.rnn(src)[0])))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DPTransformer(DPBase):
    '''
    Dual-path Transformer
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.debug = kwargs.pop('debug', False)
        num_layers = kwargs.pop('num_layers', 4)
        encoder_dim = kwargs.pop('encoder_dim', 64)

        kwargs.update({
            'd_model': encoder_dim,
            'nhead': kwargs.pop('nhead', 4),
            'dim_feedforward': kwargs.pop('dim_feedforward', 128),
            'batch_first': True
        })
        for _ in range(num_layers):
            self.row_rnn.append(ImprovedTransformedLayer(**kwargs))
            self.col_rnn.append(ImprovedTransformedLayer(**kwargs))
            self.row_norm.append(nn.GroupNorm(1, encoder_dim, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, encoder_dim, eps=1e-8))

