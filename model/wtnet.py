import torch
import torch.nn as nn
from einops import reduce
from pytorch_wavelets import DWT1D
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



class WtNet(nn.Module):

    def __init__(self, wavelet="db4", level=4, axis=2, in_seq=7, hid_seq=512, out_seq=7,in_chn=7,
                 activ="gelu", drop=0.0, pred_len=96,seq_len=192, filter_len=8, use_channel_attn=True,attn_type = "efficient"):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.in_seq = in_seq
        self.hid_seq = hid_seq
        self.out_seq = out_seq
        self.activ = activ
        self.drop = drop
        self.axis = axis
        self.mlp_block_list = nn.ModuleList()
        self.seq_len = seq_len
        self.feature = []
        # self.norm = norm

        for i in range(level):
            self.seq_len = (self.seq_len + filter_len - 1) // 2
            self.feature.append(self.seq_len)
            if i == level - 1:
                self.feature.append(self.seq_len)
        self.feature.reverse()

        self.feature_mlp = MLPBlock(1, self.out_seq, self.hid_seq, self.out_seq, self.activ, self.drop)
        # self.channel_mlp = MLPBlock(2, 11, self.hid_chn, self.in_chn, self.activ, self.drop)
        for i in range(self.level + 1):
            self.mlp_block_list.append(MLPBlock(1, self.feature[i], hid_seq, out_seq, activ, drop))
        print(f'level={level},pred_len={pred_len},filter_len={filter_len}')
    def forward(self, x, x_time):
        # x : b l c
        p_head_cpu = x.cpu().detach().numpy()
        coeffs = pywt.wavedec(p_head_cpu, wavelet=self.wavelet, level=self.level, axis=self.axis)
        coeffs_list = [torch.from_numpy(c).to(x.device) for c in coeffs]

        # coeffs = DWT1D(wave=self.wavelet, J=self.level).to(x.device)(x)

        pred_list = []
        for i, mlp_block in enumerate(self.mlp_block_list):
            pred = mlp_block(coeffs_list[i]).float()
            pred_list.append(pred)
        output = reduce(torch.stack(pred_list, 0), "h b l c -> b l c", "sum")
        y_pred = self.feature_mlp(output)
        return y_pred, x


