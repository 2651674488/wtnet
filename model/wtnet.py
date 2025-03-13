import torch
import torch.nn as nn
from einops import reduce

from utils.tools import get_activation
from utils.tools import get_dim
import pywt


class MLPBlock(nn.Module):

    def __init__(self, in_feature, hid_feature, out_feature, activ="relu", drop: float = 0.0):
        super().__init__()
        # self.dim = dim
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
        x = torch.transpose(x, 1, 2)
        x = self.net(x)
        x = torch.transpose(x, 1, 2)
        return x


class PredHead(nn.Module):
    def __init__(self, in_chn, hid_chn, activ="relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_chn, hid_chn)
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.net(x)
        x = torch.transpose(x, 1, 2)
        return x


class WtNet(nn.Module):

    def __init__(self, wavelet="db4", level=4, axis=2, in_chn=7, hid_chn=512, out_chn=7, activ="gelu", drop=0.0,
                 pred_len=96, filter_len=8):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.in_chn = in_chn
        self.hid_chn = hid_chn
        self.out_chn = out_chn
        self.activ = activ
        self.drop = drop
        self.axis = axis
        self.mlp_block_list = nn.ModuleList()
        self.feature = []
        # self.norm = norm

        for i in range(level):
            pred_len = int((pred_len + filter_len - 1) / 2)
            self.feature.append(pred_len)
            if i == level - 1:
                self.feature.append(pred_len)
        self.feature.reverse()

        # 升维
        self.pref_head = PredHead(self.in_chn, self.hid_chn)

        self.end_mlp = MLPBlock(self.out_chn, self.hid_chn, out_chn, self.activ, self.drop)
        for i in range(self.level + 1):
            self.mlp_block_list.append(MLPBlock(self.feature[i], hid_chn, out_chn, activ, drop))

    def forward(self, x, x_time):
        p_head_cpu = x.cpu().detach().numpy()
        coeffs = pywt.wavedec(p_head_cpu, wavelet=self.wavelet, level=self.level, axis=self.axis)
        coeffs_list = [torch.from_numpy(c).to(x.device) for c in coeffs]
        pred_list = []
        for i, mlp_block in enumerate(self.mlp_block_list):
            pred = mlp_block(coeffs_list[i]).float()
            pred_list.append(pred)
        output = reduce(torch.stack(pred_list, 0), "h b l c -> b l c", "sum")
        output = self.end_mlp(output)
        return output, x
