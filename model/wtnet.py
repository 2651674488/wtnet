import torch
import torch.nn as nn
from einops import reduce
from pytorch_wavelets import DWT1D, DWT1DForward
from utils.tools import get_activation
import pywt


class MLPBlock(nn.Module):

    def __init__(self,dim, in_feature, hid_feature, out_feature, activ="relu", drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.in_feature = in_feature
        self.hid_feature = hid_feature
        self.out_feature = out_feature
        self.activ = activ
        self.drop = drop
        self.net = nn.Sequential(
            nn.Linear(self.in_feature, self.hid_feature),
            get_activation(self.activ),
            nn.LayerNorm(self.hid_feature),
            nn.Dropout(self.drop),
            nn.Linear(self.hid_feature, self.out_feature),
        )


    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        x = self.net(x)
        x = torch.transpose(x, self.dim, -1)
        return x

class TimeBlock(nn.Module):
    def __init__(self,dim, in_chn, out_chn, activ="relu", drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.activ = activ
        self.net = nn.Sequential(
            nn.Linear(self.in_chn, self.out_chn),
            get_activation(self.activ),
            nn.LayerNorm(self.out_chn)
        )

    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        x = self.net(x)
        x = torch.transpose(x, self.dim, -1)
        return x


class WtNet(nn.Module):

    def __init__(self, wavelet="db4", level=4, axis=2, in_seq=7, hid_seq=512, out_seq=7,in_chn=7,
                 activ="gelu", drop=0.0, pred_len=96,seq_len=192, filter_len=8, use_channel_attn=True,attn_type = "efficient"):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.hid_seq = hid_seq
        self.out_seq = out_seq
        self.activ = activ
        self.drop = drop
        self.axis = axis
        self.mlp_block_list = nn.ModuleList()
        self.seq_len = seq_len
        self.feature = []
        self.time_feature = TimeBlock(1,4,in_chn).float()
        self.coeffs = DWT1DForward(J=self.level, wave=self.wavelet).float()
        # self.norm = norm

        for i in range(level):
            self.seq_len = (self.seq_len + filter_len - 1) // 2
            self.feature.append(self.seq_len)
            if i == level - 1:
                self.feature.append(self.seq_len)
        self.feature.reverse()

        self.feature_mlp = MLPBlock(2, self.out_seq, self.hid_seq, self.out_seq, self.activ, self.drop).float()
        for i in range(self.level + 1):
            self.mlp_block_list.append(MLPBlock(2, self.feature[i], hid_seq, out_seq, activ, drop).float())
        print(f'level={level},pred_len={pred_len},filter_len={filter_len}')

    def forward(self, x, x_time):
        # x : b l c
        x = x.float()
        x_time = x_time.float()
        x_pred = x.permute(0, 2, 1) # x:b c l
        x_time = x_time.permute(0, 2, 1) # x_time: b c l
        if x_time is not None:
            x_time = self.time_feature(x_time)
            x_pred = x_pred + x_time
        low, highs = self.coeffs(x_pred)
        coeffs_list = [low] + highs[::-1]
        pred_list = []
        for i, mlp_block in enumerate(self.mlp_block_list):
            pred = mlp_block(coeffs_list[i]).float()
            pred_list.append(pred)
        output = reduce(torch.stack(pred_list, 0), "h b l c -> b l c", "sum")
        y_pred = self.feature_mlp(output)
        y_pred = y_pred.permute(0, 2, 1)
        return y_pred, x


