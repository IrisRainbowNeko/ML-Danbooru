import torch
from torch import nn
from typing import Optional, List, Union
from einops import repeat, rearrange

from timm.models import create_model

from .metaformer_baselines import MetaFormer
from .position_encoding import build_position_encoding
from .ms_decoder import MSDecoder, MSDecoderLayer


class ML_MetaFormer(nn.Module):
    def __init__(self, encoder: MetaFormer, decoder: MSDecoder, num_queries=50, d_model=512, num_classes=1000, scale_skip=0):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.cls_head = nn.Linear(d_model, num_classes)
        self.feats_trans=nn.ModuleList([nn.Sequential(
                                            nn.LayerNorm(dim),
                                            nn.Linear(dim, d_model),
                                            nn.LayerNorm(d_model),
                                        ) for dim in self.encoder.scale_dims[scale_skip:]])
        self.pos_encoder = build_position_encoding('sine', d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.num_classes=num_classes
        self.num_queries=num_queries
        self.d_model=d_model
        self.scale_skip=scale_skip

    def encode(self, x):
        feat_list=[]
        for i in range(self.encoder.num_stage):
            x = self.encoder.downsample_layers[i](x)
            x = self.encoder.stages[i](x)
            if i>=self.scale_skip:
                feat_list.append(x)
        return feat_list

    def decode(self, feat_list):
        q = repeat(self.query_embed.weight, 'q c -> b q c', b=feat_list[0].shape[0])
        pos_emb = [rearrange(self.pos_encoder(x), 'b h w c -> b (h w) c') for x in feat_list]
        feat_list = [trans(rearrange(x, 'b h w c -> b (h w) c')) for x,trans in zip(feat_list, self.feats_trans)]

        out = self.decoder(q, feat_list, pos=pos_emb)
        return out

    def forward(self, x):
        feat_list = self.encode(x)
        pred = self.decode(feat_list) # [B, Nq_scale, L]

        pred = self.cls_head(pred)
        pred = (pred * torch.softmax(pred, dim=1)).sum(dim=1)
        return pred

def build_caformer(model_name, args):
    create_model_args = dict(
        model_name=model_name,
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=args.drop_rate if hasattr(args, 'drop_rate') else 0.0,
        drop_connect_rate=None,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path_rate if hasattr(args, 'drop_path_rate') else 0.0,
        drop_block_rate=None,
        global_pool=None,
        bn_momentum=None,
        bn_eps=None,
        scriptable=False,
        checkpoint_path=None
    )

    d_model = args.decoder_embedding
    encoder: MetaFormer = create_model(**create_model_args)
    dec_layer = MSDecoderLayer(d_model, args.num_head_decoder)
    decoder = MSDecoder(dec_layer, args.num_layers_decoder, norm=nn.LayerNorm(d_model))

    if hasattr(args, 'base_ckpt') and args.base_ckpt is not None:
        encoder.load_state_dict(torch.load(args.base_ckpt))

    model = ML_MetaFormer(encoder, decoder, num_queries=args.num_queries, num_classes=args.num_classes,
                          d_model=d_model, scale_skip=args.scale_skip)

    return model