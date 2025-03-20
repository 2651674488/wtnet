import torch
import torch.nn as nn
from einops import reduce

from utils.tools import get_activation
from utils.tools import get_dim
import pywt
import numpy as np



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
            nn.Linear(self.hid_feature, self.out_feature)
        )

    def forward(self, x):
        new_x = torch.transpose(x, 1, 2)
        new_x = self.net(new_x)
        new_x = torch.transpose(new_x, 1, 2)
        return new_x + x


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
        self.mlp_approx_block = MLPBlock(pred_len, hid_chn, out_chn, activ, drop)
        self.mlp_detail_block = MLPBlock(pred_len, hid_chn, out_chn, activ, drop)
        self.end_mlp = MLPBlock(pred_len, hid_chn, out_chn, activ, drop)
        # self.norm = norm
        self.mlp_approx_end = MLPBlock(self.out_chn, self.hid_chn, out_chn, self.activ, self.drop)
        self.mlp_detail_end = MLPBlock(self.out_chn, self.hid_chn, out_chn, self.activ, self.drop)

    def forward(self, x, x_time):
        p_head_cpu = x.cpu().detach().numpy()
        coeffs = pywt.swt(p_head_cpu, wavelet=self.wavelet, level=self.level, axis=self.axis)
        coeffs_list = [torch.from_numpy(np.array(c)).to(x.device) for c in coeffs]
        pred_list = []
        pred_approx_list = []
        pred_detail_list = []
        for i in range(len(coeffs)):
            approx, detail = coeffs_list[i]  # 解包近似系数和细节系数
            approx = self.mlp_approx_block(approx)
            detail = self.mlp_detail_block(detail)
            pred_approx_list.append(approx)
            pred_detail_list.append(detail)
        approx_all = reduce(torch.stack(pred_approx_list, 0), "h b l c -> b l c", "sum")
        detail_all = reduce(torch.stack(pred_detail_list, 0), "h b l c -> b l c", "sum")
        end = self.mlp_approx_end(approx_all) + self.mlp_detail_end(detail_all)
        output = self.end_mlp(end)
        return output, x
