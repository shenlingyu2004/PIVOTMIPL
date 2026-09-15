#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataloader import (
    HDF5BagBackend,
    LazyMIPLDataset,
    load_idx_mat,
    mipl_collate_fn,
)
from model import PIVOTMIPL
from utils import parse_args, seed_everything


def make_loader(dataset, shuffle, args, seed):
    options = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=mipl_collate_fn,
        pin_memory=args.pin_memory,
        generator=torch.Generator().manual_seed(seed),
    )
    if args.num_workers > 0:
        options.update(
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )
    return DataLoader(**options)


def cuda_sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train_one_epoch(model, optimizer, loader, dataset, device, args):
    model.train()

    for data, partial, _, local_index, _, _ in loader:
        data = data.to(
            device, non_blocking=args.pin_memory
        ).contiguous()
        partial = partial.to(device, non_blocking=args.pin_memory)

        optimizer.zero_grad(set_to_none=True)
        loss, new_partial, _ = model.calculate_objective(data, partial)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        dataset.update_partial_label(
            int(local_index), new_partial.detach().cpu()
        )


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    predictions = []
    truths = []

    for data, _, true_label, _, _, _ in loader:
        data = data.to(
            device, non_blocking=args.pin_memory
        ).contiguous()
        prediction = int(
            model.evaluate_objective(data).argmax(dim=1).item()
        )
        predictions.append(prediction)
        truths.append(int(true_label.item()))

    prediction_tensor = torch.tensor(predictions, dtype=torch.long)
    truth_tensor = torch.tensor(truths, dtype=torch.long)

    acc = prediction_tensor.eq(truth_tensor).float().mean().item()

    recalls = []
    for class_id in truth_tensor.unique(sorted=True):
        class_mask = truth_tensor.eq(class_id)
        recalls.append(
            prediction_tensor[class_mask].eq(class_id).float().mean()
        )
    bacc = torch.stack(recalls).mean().item()

    return acc, bacc


def append_metric_row(csv_path, fold, epoch, test_acc, test_bacc, train_seconds):
    file_exists = csv_path.exists()
    with csv_path.open('a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'fold',
                'epoch',
                'test_acc',
                'test_bacc',
                'train_seconds_excluding_evaluate',
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'fold': fold,
            'epoch': epoch,
            'test_acc': f'{test_acc:.6f}',
            'test_bacc': f'{test_bacc:.6f}',
            'train_seconds_excluding_evaluate': f'{train_seconds:.6f}',
        })


def run_fold(fold, backend, args, device, metrics_csv_path):
    train_ids, test_ids = load_idx_mat(
        Path(args.index_path) / f'index{fold}.mat'
    )

    train_set = LazyMIPLDataset(backend, train_ids, True)
    test_set = LazyMIPLDataset(backend, test_ids, False)
    train_loader = make_loader(
        train_set, True, args, args.seed + fold * 17
    )
    test_loader = make_loader(
        test_set, False, args, args.seed + fold * 10007
    )

    seed_everything(args.seed + fold)
    model = PIVOTMIPL(args.nr_fea, args.nr_class, args).to(device)
    optimizer = torch.optim.AdamW(
        [
            {'params': [model.prototypes], 'lr': args.lr},
            {
                'params': model.temporal_encoder.parameters(),
                'lr': args.lr * args.temporal_lr_scale,
            },
        ],
        weight_decay=args.reg,
    )

    total_train_seconds = 0.0

    for epoch in range(1, args.epochs + 1):
        cuda_sync(device)
        train_start = time.perf_counter()
        train_one_epoch(
            model, optimizer, train_loader, train_set, device, args
        )
        cuda_sync(device)
        total_train_seconds += time.perf_counter() - train_start

        if (
            epoch == 1
            or epoch % args.eval_interval == 0
            or epoch == args.epochs
        ):
            test_acc, test_bacc = evaluate(
                model, test_loader, device, args
            )
            append_metric_row(
                metrics_csv_path,
                fold,
                epoch,
                test_acc,
                test_bacc,
                total_train_seconds,
            )


def run():
    args = parse_args()
    if args.no_cuda or not torch.cuda.is_available():
        raise RuntimeError(
            'PIVOTMIPL uses official CUDA Mamba-2 kernels and '
            'requires an available NVIDIA GPU'
        )

    device = torch.device('cuda')
    seed_everything(args.seed)

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = exp_dir / 'metrics.csv'
    if metrics_csv_path.exists():
        metrics_csv_path.unlink()

    backend = HDF5BagBackend(
        args.mat_path,
        args.nr_fea,
        args.nr_class,
        args.normalize,
    )

    folds = list(range(1, 6)) if args.fold == 0 else [args.fold]
    for fold in folds:
        run_fold(fold, backend, args, device, metrics_csv_path)


if __name__ == '__main__':
    run()
