import numpy as np
import os
import random
import argparse
from tqdm import tqdm

fix_seed = 2026
random.seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, default="N15_our", help="Dataset name")
args = parser.parse_args()

in_path = os.path.join("../npz_dataset", f"{args.dataset}")
in_file = os.path.join(in_path, "valid.npz")

data = np.load(in_file)
X = data["X"]
y = data["y"]
feat_length = X.shape[1]

for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    out_file = os.path.join(in_path, f"valid_{p}.npz")
    print(f"Generating the page loaded {p}% of traffic")

    cur_X, cur_y = [], []

    for idx in tqdm(range(X.shape[0])):
        tmp_X = X[idx]
        loading_time = tmp_X[:, 0].max()
        threshold = loading_time * p / 100


        keep_idx = np.where(tmp_X[:, 0] <= threshold)[0]
        trimmed = tmp_X[keep_idx]

        pad_len = feat_length - trimmed.shape[0]
        padded = np.pad(trimmed,
                        ((0, pad_len), (0, 0)),
                        'constant', constant_values=0)
        cur_X.append(padded[np.newaxis, ...])
        cur_y.append(y[idx])

    cur_X = np.concatenate(cur_X, axis=0)
    cur_y = np.array(cur_y)
    print(f"Shape: X = {cur_X.shape}, y = {cur_y.shape}")
    np.savez(out_file, X=cur_X, y=cur_y)