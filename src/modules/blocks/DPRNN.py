import torch.nn as nn       

# from https://github.com/yluo42/TAC/blob/master/utility/models.py

class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.
    
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. 
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False, **kwargs):
        super(SingleRNN, self).__init__()
        
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.quant = kwargs.get('quant')
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size*self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        if not self.quant:
            self.rnn.flatten_parameters() # UserWarning: RNN module weights are not part of single contiguous chunk of memory
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output
    
# dual-path RNN
class DPRNN(nn.Module):
    """
    Deep duaL-path RNN.
    
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        encoder_dim: int, dimension of the input feature. The input should have shape 
                    (batch, seq_len, encoder_dim).
        hidden_size: int, dimension of the hidden state. 
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """
    def __init__(self, encoder_dim, rnn_type='LSTM', hidden_size=None, output_size=None, 
                 dropout=0, num_layers=1, bidirectional=True, **kwargs):
        super(DPRNN, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.output_size = encoder_dim if output_size is None else output_size
        self.hidden_size = encoder_dim if hidden_size is None else hidden_size        
        
        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_rnn.append(SingleRNN(rnn_type, encoder_dim, self.hidden_size, dropout, bidirectional=True, **kwargs))  # intra-segment RNN is always noncausal
            self.col_rnn.append(SingleRNN(rnn_type, encoder_dim, self.hidden_size, dropout, bidirectional=bidirectional, **kwargs))
            self.row_norm.append(nn.GroupNorm(1, encoder_dim, eps=1e-8))
            # default is to use noncausal LayerNorm for inter-chunk RNN. For causal setting change it to causal normalization techniques accordingly.
            self.col_norm.append(nn.GroupNorm(1, encoder_dim, eps=1e-8))

        # output layer    
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(encoder_dim, self.output_size, 1))
            
    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = output.permute(0,3,2,1).contiguous().view(batch_size*dim2, dim1, -1)  # B*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0,3,2,1).contiguous()  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output
            
            col_input = output.permute(0,2,3,1).contiguous().view(batch_size*dim1, dim2, -1)  # B*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0,3,1,2).contiguous()  # B, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output
            
        output = self.output(output)
            
        return output