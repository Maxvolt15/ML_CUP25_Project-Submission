import csv
import json
import os
from typing import List

import numpy as np


DEFAULT_HEADER: List[str] = [
    'time',
    'seed',
    'fold',
    'quick',
    'mode',
    'val_mee',
    'train_time_s',
    'train_samples',
    'val_samples',
    'params',
]
LEGACY_HEADER: List[str] = ['time', 'seed', 'quick', 'val_mee', 'train_time_s', 'params']


def _upgrade_legacy_results(csv_path: str) -> None:
    with open(csv_path, newline='') as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header = rows[0]
    if header == DEFAULT_HEADER:
        return
    if header != LEGACY_HEADER:
        # Unknown schema; leave untouched.
        return

    converted = [DEFAULT_HEADER]
    for row in rows[1:]:
        mapping = dict(zip(header, row))
        converted.append(
            [
                mapping.get('time'),
                mapping.get('seed'),
                '',
                mapping.get('quick'),
                'holdout',
                mapping.get('val_mee'),
                mapping.get('train_time_s'),
                '',
                '',
                mapping.get('params'),
            ]
        )

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(converted)


def append_result(csv_path, result_dict):
    """Append a result row (dictionary) to CSV, creating/migrating headers when needed."""

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    write_header = not os.path.exists(csv_path)
    if not write_header:
        _upgrade_legacy_results(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(DEFAULT_HEADER)

        row = [
            result_dict.get('time'),
            result_dict.get('seed'),
            result_dict.get('fold'),
            int(bool(result_dict.get('quick'))),
            result_dict.get('mode'),
            result_dict.get('val_mee'),
            result_dict.get('train_time_s'),
            result_dict.get('train_samples'),
            result_dict.get('val_samples'),
            json.dumps(result_dict.get('params')),
        ]
        writer.writerow(row)


def save_model_params(params, path):
    """Save model parameters (weights/biases and metadata) to a .npz file.

    params is the output of NeuralNetworkV2.get_params()
    """
    # Prepare a serializable mapping: arrays to numpy arrays
    np_params = {}
    for i, w in enumerate(params.get('weights', [])):
        np_params[f'weight_{i}'] = np.asarray(w)
    for i, b in enumerate(params.get('biases', [])):
        np_params[f'bias_{i}'] = np.asarray(b)

    # Save metadata
    np_params['layer_sizes'] = np.array(params.get('layer_sizes', []), dtype=np.int32)
    np_params['hidden_activation'] = np.array(params.get('hidden_activation', ''), dtype=object)
    np_params['output_activation'] = np.array(params.get('output_activation', ''), dtype=object)

    # Use numpy savez to persist
    np.savez_compressed(path, **np_params)
