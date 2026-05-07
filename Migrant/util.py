from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F
import numpy as np
import sys
import math


def gen_one_hot(arr, num_classes):
    binary = np.zeros((arr.shape[0], num_classes))
    for i in range(arr.shape[0]):
        binary[i, arr[i]] = 1

    return binary


def compute_metric(y_true_fine, y_pred_fine):
    y_true_fine = y_true_fine.reshape(-1, y_true_fine.shape[-1])
    y_pred_fine = y_pred_fine.reshape(-1, y_pred_fine.shape[-1])

    num_classes = np.max(y_true_fine) + 1
    y_true_fine = gen_one_hot(y_true_fine, num_classes)
    y_pred_fine = gen_one_hot(y_pred_fine, num_classes)

    result = measurement(y_true_fine, y_pred_fine, eval_metrics="Accuracy Precision Recall F1-score")
    return result


def pad_sequence(sequence, length):
    if len(sequence) >= length:
        sequence = sequence[:length]
    else:
        sequence = np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)

    return sequence


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()


def process_TW(packet_length, time, win):
    max_matrix_len = 1800
    maximum_load_time = 120

    t_arr = np.asarray(time, dtype=float)
    d_arr = np.asarray(packet_length, dtype=int)

    if t_arr.size == 0:
        N = int(math.ceil(1.0 / win))
        return np.zeros((4, N), dtype=np.float32)

    t_min, t_max = t_arr.min(), t_arr.max()
    t_norm = (t_arr - t_min) / (t_max - t_min + 1e-8)

    N = int(math.ceil(1.0 / win))
    bins = np.linspace(0, 1, N + 1, dtype=np.float32)
    up = np.zeros(N, dtype=int)
    dn = np.zeros(N, dtype=int)

    for i in range(N):
        mask = (t_norm >= bins[i]) & (t_norm < bins[i + 1])
        up[i] = d_arr[mask].clip(min=0).sum()
        dn[i] = (-d_arr[mask]).clip(min=0).sum()

    starts = bins[:-1]
    ends = bins[1:]
    return np.vstack([starts, ends, up, dn])

def measurement(y_true, y_pred, eval_metrics):
    results = {}
    for eval_metric in eval_metrics:
        if eval_metric == "Accuracy":
            results[eval_metric] = round(accuracy_score(y_true, y_pred) * 100, 2)
        elif eval_metric == "Precision":
            results[eval_metric] = round(precision_score(y_true, y_pred, average="macro") * 100, 2)
        elif eval_metric == "Recall":
            results[eval_metric] = round(recall_score(y_true, y_pred, average="macro") * 100, 2)
        elif eval_metric == "F1-score":
            results[eval_metric] = round(f1_score(y_true, y_pred, average="macro") * 100, 2)
        else:
            raise ValueError(f"Metric {eval_metric} is not matched.")
    return results


def knn_monitor(net, device, memory_data_loader, test_data_loader, num_classes, k=200, t=0.1):
    net.eval()
    total_num = 0
    feature_bank, feature_labels = [], []
    y_pred = []
    y_true = []

    with torch.no_grad():
        # Generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.to(device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous().to(device)
        feature_labels = torch.cat(feature_labels, dim=0).t().contiguous().to(device)

        # Loop through test data to predict the label by weighted kNN search
        for data, target in test_data_loader:
            data, target = data.to(device), target.to(device)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, num_classes, k, t)
            total_num += data.size(0)
            y_pred.append(pred_labels[:, 0].cpu().numpy())
            y_true.append(target.cpu().numpy())

    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()

    return y_true, y_pred


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):

    feature_labels = feature_labels.long()

    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)


    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)


    return pred_labels
