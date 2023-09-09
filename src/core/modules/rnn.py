import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int,
                 dropout: float,
                 bidirectional: bool = True):
        super(LSTM, self).__init__()

        if n_layers == 1:
            dropout = 0.0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True)

    def forward(self, xs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        total_length = xs.size(1)

        # sort by lengths
        sorted_lengths, perm_idx = torch.sort(lengths.detach().cpu(),
                                              0,
                                              descending=True)
        sorted_xs = xs[perm_idx]

        # pack
        sorted_packed_xs = pack_padded_sequence(sorted_xs,
                                                sorted_lengths,
                                                batch_first=True)
        # forward
        sorted_packed_ys, (sorted_hs, _) = self.lstm(sorted_packed_xs)

        # unpack
        sorted_ys, _ = pad_packed_sequence(sorted_packed_ys,
                                           batch_first=True,
                                           total_length=total_length)
        # unsort
        perm_idx_rev = self._inverse_indices(perm_idx)
        ys = sorted_ys[perm_idx_rev, :, :]
        return ys

    def _inverse_indices(self, indices):
        r = torch.empty_like(indices)
        r[indices] = torch.arange(len(indices))
        return r


if __name__ == '__main__':
    # build rnn
    input_size = 16
    hidden_size = 64
    num_layers = 2
    dropout = 0.1
    bidirectional = True
    rnn = LSTM(input_size, hidden_size, num_layers, dropout, bidirectional)
    print(rnn)

    # input
    BATCH_SIZE = 10
    MAX_LENGTH = 32
    xs = torch.rand(BATCH_SIZE, MAX_LENGTH, input_size)
    lengths = torch.randint(1, MAX_LENGTH, (BATCH_SIZE, ))

    # set device
    device = torch.device('cuda')
    rnn.to(device)
    xs = xs.to(device)
    lengths = lengths.to(device)

    # forward
    print(xs.size())
    print(lengths)
    # ys, hs = rnn(xs, lengths)
    # print(ys.size(), hs.size())
    ys = rnn(xs, lengths)
    print(ys.size())
