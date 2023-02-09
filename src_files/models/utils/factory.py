import logging
import os
from urllib import request

import torch

from ...ml_decoder import ml_decoder_colo, ml_decoder
from ..tresnet import tresnet_f, tresnet

logger = logging.getLogger(__name__)

def create_model(args, load_head=False, colo=False):
    """Create a model
    """
    tres = tresnet_f if args.frelu else tresnet
    mld = ml_decoder_colo if colo else ml_decoder

    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name == 'tresnet_m':
        model = tres.TResnetM(model_params)
    elif args.model_name == 'tresnet_d':
        model = tres.TResnetD(model_params)
    elif args.model_name == 'tresnet_l':
        model = tres.TResnetL(model_params)
    elif args.model_name == 'tresnet_xl':
        model = tres.TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    ####################################################################################
    if args.use_ml_decoder:
        model = mld.add_ml_decoder_head(model,num_classes=args.num_classes,num_of_groups=args.num_of_groups,
                                    decoder_embedding=args.decoder_embedding, zsl=args.zsl, use_xformers=args.xformers,
                                        learn_query=args.learn_query if hasattr(args, 'learn_query') else False)
    ####################################################################################
    # loading pretrain model
    model_path = args.model_path
    if args.model_name == 'tresnet_l' and os.path.exists("./tresnet_l.pth"):
        model_path = "./tresnet_l.pth"
    if model_path:  # make sure to load pretrained model
        if not os.path.exists(model_path):
            print("downloading pretrain model...")
            request.urlretrieve(args.model_path, "./tresnet_l.pth")
            model_path = "./tresnet_l.pth"
            print('done')
        state = torch.load(model_path, map_location='cpu')
        if not load_head:
            if 'model' in state:
                key = 'model'
            else:
                key = 'state_dict'
            filtered_dict = {k: v for k, v in state[key].items() if
                             (k in model.state_dict() and ('head.fc' not in k)
                              and ('head.decoder.duplicate_pooling' not in k)
                              and ('head.decoder.duplicate_pooling_bias' not in k)
                              and ('head.decoder.query_embed.weight' not in k)
                              )}
            model.load_state_dict(filtered_dict, strict=False)
        else:
            model.load_state_dict(state[key], strict=True)

    return model
