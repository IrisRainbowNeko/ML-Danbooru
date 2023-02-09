import torch
from torch import nn
import torch.nn.functional as F
import math

from .layer import *

def add_ml_decoder_head(model, num_classes=-1, num_of_groups=-1, decoder_embedding=768, zsl=0, use_xformers=True, learn_query=False):
    if num_classes == -1:
        num_classes = model.num_classes
    num_features = model.num_features
    if hasattr(model, 'global_pool') and hasattr(model, 'fc'):  # resnet50
        model.global_pool = nn.Identity()
        del model.fc
        model.fc = MLDecoder(num_classes=num_classes, initial_num_features=num_features, num_of_groups=num_of_groups,
                             decoder_embedding=decoder_embedding, zsl=zsl, use_xformers=use_xformers, learn_query=learn_query)
    elif hasattr(model, 'head'):  # tresnet
        if hasattr(model, 'global_pool'):
            model.global_pool = nn.Identity()
        del model.head
        model.head = MLDecoder(num_classes=num_classes, initial_num_features=num_features, num_of_groups=num_of_groups,
                               decoder_embedding=decoder_embedding, zsl=zsl, use_xformers=use_xformers, learn_query=learn_query)
    else:
        print("model is not suited for ml-decoder")
        exit(-1)

    return model


#@torch.jit.script
def f_groupfc(h: torch.Tensor, duplicate_pooling: nn.ParameterList, out_extrap: torch.Tensor):
    for i in range(h.shape[1]):
        h_i = h[:, i, :]
        w_i = duplicate_pooling[i]
        out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class GroupFC(nn.Module):
    def __init__(self, embed_len_decoder: int, duplicate_factor: int, decoder_embedding: int, num_classes: int):
        super().__init__()

        self.embed_len_decoder = embed_len_decoder
        self.num_classes = num_classes
        self.zsl = False

        #self.duplicate_pooling = torch.nn.Parameter(torch.Tensor(embed_len_decoder, decoder_embedding * duplicate_factor))
        self.duplicate_pooling = nn.ParameterList([torch.Tensor(decoder_embedding, duplicate_factor) for _ in  range(embed_len_decoder)])
        self.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))

        self.decoder_embedding=decoder_embedding
        self.duplicate_factor=duplicate_factor

        for p in self.duplicate_pooling:
            torch.nn.init.xavier_normal_(p)
        #self.xavier_normal_embedding(self.duplicate_pooling, decoder_embedding, duplicate_factor)
        torch.nn.init.constant_(self.duplicate_pooling_bias, 0)

    def xavier_normal_embedding(self, tensor: Tensor, num_input_fmaps, num_output_fmaps, gain: float = 1.) -> Tensor:
        def _calculate_fan_in_and_fan_out(tensor):
            receptive_field_size = 1
            #receptive_field_size *= tensor.shpae[0]
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

            return fan_in, fan_out

        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

        return torch.nn.init._no_grad_normal_(tensor, 0., std)

    def forward(self, h: torch.Tensor):
        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.duplicate_factor, device=h.device, dtype=h.dtype)

        f_groupfc(h, self.duplicate_pooling, out_extrap)

        if not self.zsl:
            h_out = out_extrap.flatten(1)[:, :self.num_classes]
        else:
            h_out = out_extrap.flatten(1)
        h_out = h_out + self.duplicate_pooling_bias
        return h_out

class MLDecoder(nn.Module):
    def __init__(self, num_classes, num_of_groups=-1, decoder_embedding=768,
                 initial_num_features=2048, zsl=0, use_xformers=True, learn_query=False):
        super(MLDecoder, self).__init__()
        self.use_xformers=use_xformers
        
        embed_len_decoder = 256 if num_of_groups < 0 else num_of_groups
        if embed_len_decoder > num_classes:
            embed_len_decoder = num_classes

        # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        self.embed_standart = nn.Linear(initial_num_features, decoder_embedding)

        # non-learnable queries
        if not zsl:
            self.query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
            self.query_embed.requires_grad_(learn_query)
        else:
            self.query_embed = None

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048
        layer_decode = (TransformerDecoderLayerOptimal_XFromers if use_xformers else TransformerDecoderLayerOptimal) \
            (d_model=decoder_embedding, dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)
        self.zsl = zsl

        if self.zsl:
            if decoder_embedding != 300:
                self.wordvec_proj = nn.Linear(300, decoder_embedding)
            else:
                self.wordvec_proj = nn.Identity()
            self.duplicate_pooling = torch.nn.Parameter(torch.Tensor(decoder_embedding, 1))
            self.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))
            self.duplicate_factor = 1
        else:
            # group fully-connected
            self.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)

        self.group_fc = GroupFC(embed_len_decoder, self.duplicate_factor, decoder_embedding, num_classes)

    def forward(self, x):
        if len(x.shape) == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:  # [bs, 197,468]
            embedding_spatial = x
        embedding_spatial_786 = self.embed_standart(embedding_spatial)
        embedding_spatial_786 = F.relu(embedding_spatial_786, inplace=True)

        bs = embedding_spatial_786.shape[0]
        if self.zsl:
            query_embed = F.relu(self.wordvec_proj(self.query_embed))
        else:
            query_embed = self.query_embed.weight

        tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)  # no allocation of memory with expand
        if self.use_xformers:
            h = self.decoder(tgt, embedding_spatial_786)  # [B, N_query, 768]
        else:
            h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1))  # [N_query, B, 768]
            h = h.transpose(0, 1)  # [B, N_query, 768]

        logits = self.group_fc(h)

        return logits
