import argparse
import configparser
import numpy as np
import random
import tqdm
import torch
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import json

from Migrant import Migrant
from Migrant.Migrant import Migrant
from dataset_util import MigrantDataset,TransDataset,DirectionDataset
from util import measurement, knn_monitor, compute_metric
import warnings

warnings.filterwarnings("ignore")

# Set a fixed seed for reproducibility
fix_seed = 2026
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed(fix_seed)
np.random.seed(fix_seed)
rng = np.random.RandomState(fix_seed)
cudnn.benchmark = False
cudnn.deterministic = True

# Config
parser = argparse.ArgumentParser(description="Migrant")
parser.add_argument("--checkpoints", type=str, default="../checkpoints/", help="Location of model checkpoints")
parser.add_argument("--dataset", type=str, required=True, default="N15", help="Dataset name")
parser.add_argument("--device", type=str, default="cpu", help="Device, options=[cpu, cuda, cuda:x]")
parser.add_argument("--model", type=str, required=True, default="Migrant", help="Model name")
parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")
parser.add_argument('--window', type=str, default='1/1800', help="Window size")
parser.add_argument("--ratio", type=int, default=100, help="loaded ratio")

# Optimization parameters
parser.add_argument("--batch_size", type=int, default=64, help="Batch size of train input data")

# Output parameters
parser.add_argument('--eval_metrics', nargs='+', required=True, type=str,
                    help="Evaluation metrics, options=[Accuracy, Precision, Recall, F1-score, P@min, r-Precision]")
parser.add_argument("--save_metric", type=str, default="F1-score",
                    help="Save the model when the metric reaches its maximum value on the validation set")
parser.add_argument("--load_name", type=str, default="base", help="Name of the model file")
parser.add_argument("--result_file", type=str, default="result", help="File to save test results")
args = parser.parse_args()

if '/' in args.window:
    num, den = args.window.split('/')
    args.window = float(num) / float(den)
else:
    args.window = float(args.window)


# Preparation
device = torch.device(args.device)

in_path = os.path.join("../npz_dataset", args.dataset)
log_path = os.path.join("../logs", args.dataset, args.model)
ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
os.makedirs(log_path, exist_ok=True)
out_file = os.path.join(log_path, f"{args.result_file}.json")


# Dataset
def load_data(data_path):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    return X, y


valid_X, valid_y = load_data(os.path.join("../npz_dataset", args.dataset, f"valid.npz"))
test_X, test_y = load_data(os.path.join(in_path, f"test.npz"))

num_classes = len(np.unique(test_y))



val_set = MigrantDataset(valid_X, valid_y, args.seq_len, args.window, args.ratio)
test_dataset = MigrantDataset(test_X, test_y, args.seq_len, args.window, args.ratio)


val_loader = DataLoader(val_set, batch_size=args.batch_size,
                        shuffle=False, drop_last=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                         shuffle=False, drop_last=False, num_workers=4)

# Model
model = Migrant(num_classes=num_classes)


model.load_state_dict(torch.load(os.path.join(ckp_path, f"{args.load_name}.pth"), map_location="cpu"))
model.to(device)


valid_true, valid_pred = knn_monitor(model, device, val_loader, test_loader, num_classes, 10)
result = measurement(valid_true, valid_pred, args.eval_metrics)

print(result)
with open(out_file, "w") as fp:
    json.dump(result, fp, indent=4)
