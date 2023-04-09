import os
import torch
from torch import nn
from typing import Optional, List, Union
import copy

from .metaformer_baselines import Attention

try:
    use_xformers = os.getenv('USE_XFORMERS')
    from xformers.components import MultiHeadDispatch, build_attention
except:
    use_xformers = False
    print('xformers not find')

class MSDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=nn.GELU(), normalize_before=False):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, attn_drop=dropout, is_2d=False)
        self.cross_attn = Attention(d_model, nhead, attn_drop=dropout, is_2d=False)
        # Implementation of Feedforward model
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt: Union[torch.Tensor, List[torch.Tensor]], memory:List[torch.Tensor],
                    tgt_mask: Optional[torch.Tensor] = None,
                    memory_mask: Optional[torch.Tensor] = None,
                    pos: Optional[List[torch.Tensor]] = None,
                    query_pos: Optional[List[torch.Tensor]] = None):
        '''
            tgt: [B, Nq, L]xN_scale
            tgt: [B, Nkv, L]xN_scale
        '''
        multi_scale = isinstance(tgt, list) or isinstance(tgt, tuple)

        # self-attention
        if multi_scale:
            q_lens = [item.shape[1] for item in tgt]
            tgt = torch.cat(tgt, dim=1)
        else:
            q_lens = [tgt.shape[1]]*len(memory)
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, v=tgt2, att_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        if multi_scale:
            tgt = tgt.split(q_lens, dim=1)
        else:
            tgt = [tgt for _ in range(len(memory))]

        # cross-attention
        q_list=[]
        for q, kv, p in zip(tgt, memory, pos):
            tgt2 = self.norm2(q)
            tgt2 = self.cross_attn(q=self.with_pos_embed(tgt2, query_pos),
                                    k=self.with_pos_embed(kv, p),
                                    v=kv, att_mask=memory_mask)
            q = q + self.dropout2(tgt2)
            q_list.append(q)

        tgt = torch.cat(q_list, dim=1)
        tgt2 = self.norm3(tgt)
        tgt2 = self.ff(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        tgt = tgt.split(q_lens, dim=1)

        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MSDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(torch.cat(output, dim=1)))

        if self.norm is not None:
            output = self.norm(torch.cat(output, dim=1))
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output