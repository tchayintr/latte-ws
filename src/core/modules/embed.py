import torch
import torch.nn as nn
from typing import List, Union


class Embedding(nn.ModuleList):

    def __init__(self,
                 input_size: int,
                 embed_size: int,
                 padding_idx: int = 0,
                 dropout: float = 0.3,
                 device: str = 'cuda'):

        embed = nn.Embedding(input_size,
                             embed_size,
                             padding_idx=padding_idx,
                             device=device)
        modules = []
        modules.append(embed)
        modules.append(nn.Dropout(dropout))
        super(Embedding, self).__init__(modules)

    def forward(self, x):
        for module in self:
            x = module(x)
        return x


class BertEmbedding(nn.ModuleList):

    def __init__(self,
                 bert_model,
                 token_ids: Union[List[int], torch.Tensor],
                 embed_size: int = 768,
                 padding_idx: int = 0,
                 bert_mode: str = 'none',
                 device: str = 'cuda'):
        self.bert_model = bert_model
        self.token_ids = token_ids
        self.embed_size = embed_size
        self.padding_idx = padding_idx
        self.bert_mode = bert_mode
        self._construct_bert_embed(bert_model, token_ids, embed_size,
                                   padding_idx, bert_mode, device)

    @staticmethod
    def _construct_bert_embed(bert_model,
                              token_ids,
                              embed_size=768,
                              padding_idx=0,
                              bert_mode='none',
                              device='cuda'):
        bert_inputs = torch.tensor(token_ids, dtype=torch.long, device=device)
        outputs = bert_model(bert_inputs)
        if bert_mode == 'none':
            outputs = outputs[0]
        elif bert_mode == 'concat':
            outputs = outputs[2][-4:]
            outputs = torch.cat(outputs, dim=-1)
        elif bert_mode == 'sum':
            outputs = outputs[2][-4:]
            outputs = torch.stack(outputs, dim=0).sum(dim=0)
        elif bert_mode == 'sum-all':
            outputs = outputs[2][:]
            outputs = torch.stack(outputs, dim=0).sum(dim=0)

        embed = nn.Embedding.from_pretrained(outputs,
                                             freeze=True,
                                             padding_idx=padding_idx)
        return embed
