import torch

import argparse

from NeuFlow.neuflow import NeuFlow
from data_utils.evaluate import validate_things, validate_sintel, validate_kitti

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrain model for finetuing or resume from terminated training')

    return parser

def main(args):
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda')

    model = NeuFlow().to(device)

    checkpoint = torch.load(args.resume, map_location='cuda')

    model.load_state_dict(checkpoint['model'], strict=False)

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)

    # validate_things(model, dstype='frames_cleanpass', validate_subset=True, max_val_flow=400)
    # validate_things(model, dstype='frames_finalpass', validate_subset=False, max_val_flow=400)
    # validate_sintel(model, dstype='clean')
    # validate_sintel(model, dstype='final')
    validate_kitti(model)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
