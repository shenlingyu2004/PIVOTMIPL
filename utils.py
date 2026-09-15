#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import random
from pathlib import Path

import numpy as np
import torch


CLIP_FEATURE_KEYS = ('slowfast', 'mae')
FRAME_FEATURE_KEYS = ('dinov3', 'resnet')


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1', 'yes'}:
        return True
    if value.lower() in {'false', '0', 'no'}:
        return False
    raise argparse.ArgumentTypeError('expected True or False')


def infer_temporal_branch(mat_path):
    basename = Path(mat_path).name.lower()
    clip_hits = [key for key in CLIP_FEATURE_KEYS if key in basename]
    frame_hits = [key for key in FRAME_FEATURE_KEYS if key in basename]
    if clip_hits and frame_hits:
        raise ValueError(
            f'ambiguous feature type in MAT basename {basename!r}: '
            f'clip keys={clip_hits}, frame keys={frame_hits}'
        )
    if clip_hits:
        return 'clip', '+'.join(clip_hits)
    if frame_hits:
        return 'frame', '+'.join(frame_hits)
    raise ValueError(
        f'cannot infer temporal branch from MAT basename {basename!r}; '
        f'expected one of {CLIP_FEATURE_KEYS + FRAME_FEATURE_KEYS}'
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='PIVOTMIPL Renyi-Occupancy Plan-Momentum MOTE'
    )
    parser.add_argument('--mat_path', required=True)
    parser.add_argument('--index_path', required=True)
    parser.add_argument('--nr_fea', type=int, required=True)
    parser.add_argument('--nr_class', type=int, required=True)
    parser.add_argument('--normalize', type=str2bool, default=False)

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument(
        '--fold', type=int, default=0,
        help='fold 1-5 runs one fold; 0 runs all five folds',
    )
    parser.add_argument('--no_cuda', action='store_true')

    # ROPM-MOTE. Numerical solver controls do not define extra losses.
    parser.add_argument('--num_prototypes', type=int, default=12)
    parser.add_argument('--ot_eps', type=float, default=0.07)
    parser.add_argument('--occupancy_weight', type=float, default=0.01)
    parser.add_argument('--time_root_iters', type=int, default=12)
    parser.add_argument('--teacher_projection_iters', type=int, default=120)
    parser.add_argument('--plan_tol', type=float, default=1e-6)
    parser.add_argument('--geometry_eps', type=float, default=1e-7)
    parser.add_argument('--label_momentum', type=float, default=0.9)

    # PureMIPLv91 temporal encoder, unchanged.
    parser.add_argument('--temporal_hidden', type=int, default=512)
    parser.add_argument('--temporal_layers', type=int, default=4)
    parser.add_argument('--temporal_ffn_expand', type=int, default=4)
    parser.add_argument('--temporal_dropout', type=float, default=0.1)
    parser.add_argument('--mamba_d_state', type=int, default=128)
    parser.add_argument('--mamba_d_conv', type=int, default=4)
    parser.add_argument('--mamba_expand', type=int, default=2)
    parser.add_argument('--mamba_headdim', type=int, default=64)
    parser.add_argument('--mamba_chunk_size', type=int, default=256)
    parser.add_argument('--motion_kernel', type=int, default=5)
    parser.add_argument('--motion_dilation_1', type=int, default=1)
    parser.add_argument('--motion_dilation_2', type=int, default=2)
    parser.add_argument('--temporal_lr_scale', type=float, default=0.1)

    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--reg', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=5.0)

    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--persistent_workers', type=str2bool, default=True)
    parser.add_argument('--pin_memory', type=str2bool, default=True)

    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--exp_dir', default='./logs_PIVOTMIPL')
    args = parser.parse_args()

    if args.nr_fea <= 0 or args.nr_class <= 1:
        parser.error('--nr_fea must be positive and --nr_class must exceed 1')
    if args.epochs <= 0:
        parser.error('--epochs must be positive')
    if args.fold < 0 or args.fold > 5:
        parser.error('--fold must be 0 or an integer from 1 to 5')
    if args.num_prototypes <= 0:
        parser.error('--num_prototypes must be positive')
    if args.ot_eps <= 0.0:
        parser.error('--ot_eps must be positive')
    if args.occupancy_weight < 0.0:
        parser.error('--occupancy_weight must be non-negative')
    if args.time_root_iters <= 0:
        parser.error('--time_root_iters must be positive')
    if args.teacher_projection_iters <= 0:
        parser.error('--teacher_projection_iters must be positive')
    if args.plan_tol <= 0.0:
        parser.error('--plan_tol must be positive')
    if not 0.0 < args.geometry_eps < 0.1:
        parser.error('--geometry_eps must be in (0, 0.1)')
    if not 0.0 <= args.label_momentum < 1.0:
        parser.error('--label_momentum must be in [0, 1)')

    if args.temporal_hidden <= 0 or args.temporal_layers <= 0:
        parser.error('temporal hidden/layers must be positive')
    if args.temporal_ffn_expand <= 0:
        parser.error('--temporal_ffn_expand must be positive')
    if not 0.0 <= args.temporal_dropout < 1.0:
        parser.error('--temporal_dropout must be in [0, 1)')
    if args.mamba_d_state <= 0 or args.mamba_d_conv <= 0:
        parser.error('Mamba state/conv dimensions must be positive')
    if args.mamba_expand <= 0 or args.mamba_headdim <= 0:
        parser.error('Mamba expand/headdim must be positive')
    if args.mamba_chunk_size <= 0:
        parser.error('--mamba_chunk_size must be positive')
    if (
        args.temporal_hidden * args.mamba_expand
        % args.mamba_headdim != 0
    ):
        parser.error(
            '--temporal_hidden * --mamba_expand must be divisible '
            'by --mamba_headdim'
        )
    if args.motion_kernel <= 0 or args.motion_kernel % 2 == 0:
        parser.error('--motion_kernel must be a positive odd integer')
    if args.motion_dilation_1 <= 0 or args.motion_dilation_2 <= 0:
        parser.error('motion dilations must be positive')
    if args.temporal_lr_scale <= 0.0:
        parser.error('--temporal_lr_scale must be positive')
    if args.lr <= 0.0 or args.reg < 0.0 or args.grad_clip <= 0.0:
        parser.error('optimizer learning rate/regularization/clip are invalid')
    if args.num_workers < 0:
        parser.error('--num_workers must be non-negative')
    if args.prefetch_factor <= 0:
        parser.error('--prefetch_factor must be positive')
    if args.eval_interval <= 0:
        parser.error('--eval_interval must be positive')

    try:
        args.temporal_branch, args.feature_key = infer_temporal_branch(
            args.mat_path
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
