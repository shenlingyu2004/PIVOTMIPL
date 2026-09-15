#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data_utils


def load_idx_mat(path):
    index = sio.loadmat(path)
    train = np.asarray(index['trainIndex'], dtype=np.int64).reshape(-1).tolist()
    test = np.asarray(index['testIndex'], dtype=np.int64).reshape(-1).tolist()
    return train, test


class HDF5BagBackend:
    def __init__(self, mat_path, nr_fea, nr_class, normalize=False):
        self.mat_path = mat_path
        self.nr_fea = nr_fea
        self.normalize = normalize
        self._h5 = None

        with h5py.File(mat_path, 'r') as file:
            table = file['data']
            self.num_bags = int(table.shape[0])
            self.feature_paths = []
            labels = np.empty(self.num_bags, dtype=np.int64)
            partial = np.zeros((self.num_bags, nr_class), dtype=np.float32)
            for bag in range(self.num_bags):
                self.feature_paths.append(file[table[bag, 0]].name)
                candidate = np.asarray(
                    file[table[bag, 1]], dtype=np.int64
                ).reshape(-1)
                partial[bag, candidate] = 1.0 / candidate.size
                labels[bag] = np.asarray(
                    file[table[bag, 2]], dtype=np.int64
                ).reshape(-1)[0]
        self.true_bag_lab = torch.from_numpy(labels)
        self.partial_bag_lab_processed = torch.from_numpy(partial)

        self.mean = None
        self.std = None
        if self.normalize:
            self._compute_normalization()

    def _compute_normalization(self, chunk_size=2048):
        total_n = 0
        total_sum = np.zeros(self.nr_fea, dtype=np.float64)
        total_sumsq = np.zeros(self.nr_fea, dtype=np.float64)
        with h5py.File(self.mat_path, 'r') as file:
            for path in self.feature_paths:
                feature = file[path]
                for start in range(0, feature.shape[0], chunk_size):
                    block = np.asarray(
                        feature[start:start + chunk_size, :], dtype=np.float32
                    ).astype(np.float64)
                    total_n += block.shape[0]
                    total_sum += block.sum(axis=0)
                    total_sumsq += (block * block).sum(axis=0)
        mean = total_sum / total_n
        std = np.sqrt(np.maximum(total_sumsq / total_n - mean * mean, 0.0))
        std[std == 0] = 1.0
        self.mean = mean.astype(np.float32).reshape(1, -1)
        self.std = std.astype(np.float32).reshape(1, -1)

    def read_bag(self, offset):
        if self._h5 is None:
            self._h5 = h5py.File(self.mat_path, 'r')
        feature = np.asarray(
            self._h5[self.feature_paths[offset]], dtype=np.float32
        )
        if self.normalize:
            feature = (feature - self.mean) / self.std
        return np.ascontiguousarray(feature)


class LazyMIPLDataset(data_utils.Dataset):
    def __init__(self, backend, idx_list, share_partial_labels):
        self.backend = backend
        self.idx_list = idx_list
        offsets = torch.as_tensor(self.idx_list, dtype=torch.long) - 1
        self.partial_bag_lab_tensor = backend.partial_bag_lab_processed[offsets]
        self.true_bag_lab_tensor = backend.true_bag_lab[offsets]
        if share_partial_labels and self.partial_bag_lab_tensor.numel():
            self.partial_bag_lab_tensor.share_memory_()

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        bag_id = self.idx_list[index]
        feature = self.backend.read_bag(bag_id - 1)
        return (
            torch.from_numpy(feature),
            self.partial_bag_lab_tensor[index].view(1, -1),
            self.true_bag_lab_tensor[index].view(1),
            index,
            bag_id,
            int(feature.shape[0]),
        )

    def update_partial_label(self, local_index, new_y):
        self.partial_bag_lab_tensor[local_index].copy_(new_y.reshape(-1))

    def candidate_labels_str(self, local_index):
        labels = torch.nonzero(
            self.partial_bag_lab_tensor[local_index] > 0,
            as_tuple=False,
        ).reshape(-1).tolist()
        return ';'.join(map(str, labels))


def mipl_collate_fn(batch):
    return batch[0]
