import argparse
import configparser
import numpy as np
import random
import tqdm
import torch
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset_util import MigrantDataset,TransDataset,DirectionDataset
from Migrant import Migrant
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

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description="Migrant")
parser.add_argument("--dataset", type=str, required=True, default="N15", help="Dataset name")
parser.add_argument("--device", type=str, default="cpu", help="Device, options=[cpu, cuda, cuda:x]")
parser.add_argument("--model", type=str, required=True, default="Migrant", help="Model name")
parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")
parser.add_argument('--window', type=str, default='1/1800', help="Window size")
parser.add_argument("--ratio", type=int, default=100, help="loaded ratio")
# Optimization parameters
parser.add_argument("--train_epochs", type=int, default=50, help="Train epochs")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size of train input data")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer learning rate")
parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer")

# Output parameters
parser.add_argument('--eval_metrics', nargs='+', required=True, type=str,
                    help="Evaluation metrics, options=[Accuracy, Precision, Recall, F1-score, P@min, r-Precision]")
parser.add_argument("--save_metric", type=str, default="F1-score",
                    help="Save the model when the metric reaches its maximum value on the validation set")
parser.add_argument("--load_file", type=str, default=None, help="The pre-trained model file")
parser.add_argument("--save_name", type=str, default="base", help="Name used to save the model")

# Parse arguments
args = parser.parse_args()

if '/' in args.window:
    num, den = args.window.split('/')
    args.window = float(num) / float(den)
else:
    args.window = float(args.window)


# Preparation
device = torch.device(args.device)

ckp_path = os.path.join("../checkpoints", args.dataset, args.model)
os.makedirs(ckp_path, exist_ok=True)
out_file = os.path.join(ckp_path, f"{args.save_name}.pth")


# Dataset
def load_data(data_path):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    return X, y

train_X, train_y = load_data(os.path.join("../npz_dataset", args.dataset, f"train.npz"))
valid_X, valid_y = load_data(os.path.join("../npz_dataset", args.dataset, f"valid.npz"))

num_classes = len(np.unique(train_y))
print(num_classes)


#data process
train_set = MigrantDataset(train_X, train_y, args.seq_len, args.window, args.ratio)
val_set = MigrantDataset(valid_X, valid_y, args.seq_len, args.window, args.ratio)


train_loader = DataLoader(train_set, batch_size=args.batch_size,
                          shuffle=True, drop_last=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=args.batch_size,
                        shuffle=False, drop_last=False, num_workers=4)

# Model
model = Migrant(num_classes=num_classes)


if args.load_file is None:
    print("No pre-trained model")
else:
    print("Loading the pretrained model in ", args.load_file)
    checkpoint = torch.load(args.load_file, map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=False)

model.to(device)
optimizer = eval(f"torch.optim.{args.optimizer}")(model.parameters(),
                                                                 lr=float(args.learning_rate),)

criterion = torch.nn.CrossEntropyLoss()


# Train
metric_best_value = 0
for epoch in range(args.train_epochs):
    model.train()
    sum_loss = 0
    sum_count = 0
    for index, cur_data in enumerate(tqdm.tqdm(train_loader)):
        cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
        optimizer.zero_grad()
        outs = model(cur_X)

        loss = criterion(outs, cur_y)

        loss.backward()
        optimizer.step()
        sum_loss += loss.data.cpu().numpy() * outs.shape[0]
        sum_count += outs.shape[0]

    train_loss = round(sum_loss / sum_count, 3)
    print(f"epoch {epoch}: train_loss = {train_loss}")


    valid_true, valid_pred = knn_monitor(model, device, train_loader, val_loader, num_classes, 10)

    valid_result = measurement(valid_true, valid_pred, args.eval_metrics)
    print(f"{epoch}: {valid_result}")

    if valid_result["F1-score"] > metric_best_value:
        metric_best_value = valid_result["F1-score"]
        torch.save(model.state_dict(), out_file)
