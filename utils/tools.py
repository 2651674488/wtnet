import re

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from pytorch_lightning.loggers import CSVLogger

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def visual(config, true, preds=None):
    """
    改进后的可视化函数，自动按数据集分类存储并防止文件覆盖
    """
    # 构建存储路径
    base_dir = os.path.join('./pic', config.name)
    os.makedirs(base_dir, exist_ok=True)

    # 生成文件名基础部分（包含关键参数）
    file_base = f"pred{config.pred_len}_{config.features}"

    # 查找现有文件的最大编号
    max_num = 0
    pattern = re.compile(rf"^{file_base}_(\d+)\.png$")
    for filename in os.listdir(base_dir):
        match = pattern.match(filename)
        if match:
            current_num = int(match.group(1))
            max_num = max(max_num, current_num)

    # 生成新文件名
    new_filename = f"{file_base}_{max_num + 1}.png"
    save_path = os.path.join(base_dir, new_filename)

    # 绘图逻辑保持不变
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.xlabel("Time Steps", fontsize=12)  # 更合理的坐标标签
    plt.ylabel("Value", fontsize=12)
    plt.title(f'{config.name} (Prediction Length: {config.pred_len})', fontsize=14)
    plt.grid(True, alpha=0.3)
    param_text = (
        f"Model Parameters:\n"
        f"• Learning Rate: {getattr(config, 'lr', 'N/A')}\n"
        f"• Dropout Rate: {getattr(config, 'drop', 'N/A')}\n"
        f"• Level: {getattr(config, 'level', 'N/A')}\n"
        f"• Batch Size: {getattr(config, 'batch_size', 'N/A')}\n"
    )

    # 在图表右下角添加参数文本框
    plt.text(0.95, 0.05, param_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round'))

    # 保存文件
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def get_csv_logger(save_dir, name):
    root_dir = os.fspath(os.path.join(save_dir, name))
    if not os.path.isdir(root_dir):
        return CSVLogger(save_dir, name)
    else:
        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir,
                                          d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return CSVLogger(save_dir, name)

        return CSVLogger(save_dir, name, max(existing_versions) + 1)


def get_activation(activ: str):
    if activ == "gelu":
        return nn.GELU()
    elif activ == "sigmoid":
        return nn.Sigmoid()
    elif activ == "tanh":
        return nn.Tanh()
    elif activ == "relu":
        return nn.ReLU()

    raise RuntimeError("activation should not be {}".format(activ))


def get_loss_fn(loss_fn: str):
    if loss_fn == "mse":
        return nn.MSELoss()
    elif loss_fn == "mae":
        return nn.L1Loss()
    elif loss_fn == "huber":
        return nn.HuberLoss(delta=1.0)

    raise RuntimeError("loss function should not be {}".format(loss_fn))



