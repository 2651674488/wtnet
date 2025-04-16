import torch
import torch.nn as nn
from einops import reduce

from utils.tools import get_activation
from utils.tools import get_dim
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
        pred_list = []
        for i, mlp_block in enumerate(self.mlp_block_list):
            pred = mlp_block(coeffs_list[i]).float()
            pred_list.append(pred)
        output = reduce(torch.stack(pred_list, 0), "h b l c -> b l c", "sum")
        y_pred = self.feature_mlp(output)
        return y_pred, x

##################################

'''
import torch
import torch.nn as nn
import pywt
from einops import reduce
from einops.layers.torch import Rearrange


class ChannelAttention(nn.Module):
    """Channel-wise self-attention mechanism"""

    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.query = nn.Linear(channels, channels // reduction_ratio)
        self.key = nn.Linear(channels, channels // reduction_ratio)
        self.value = nn.Linear(channels, channels)
        self.scale = (channels // reduction_ratio) ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch, length, channels)
        b, l, c = x.shape

        # Rearrange to (batch, length, channels) -> (batch*length, channels)
        x_reshaped = x.reshape(-1, c)

        queries = self.query(x_reshaped)  # (b*l, c/r)
        keys = self.key(x_reshaped)  # (b*l, c/r)
        values = self.value(x_reshaped)  # (b*l, c)

        # Compute attention scores (batch*length, c/r) x (batch*length, c/r) -> (batch*length, batch*length)
        attn_scores = torch.matmul(queries, keys.transpose(0, 1)) * self.scale
        attn_probs = self.softmax(attn_scores)

        # Apply attention to values (batch*length, batch*length) x (batch*length, c) -> (batch*length, c)
        out = torch.matmul(attn_probs, values)

        # Reshape back to original
        out = out.reshape(b, l, c)
        return out

class EfficientChannelAttention(nn.Module):
    """More efficient channel-wise attention"""

    def __init__(self, channels, reduction_ratio=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, length, channels)
        b, l, c = x.shape

        # Compute channel attention weights
        y = self.avg_pool(x.transpose(1, 2))  # (b, c, 1)
        y = y.view(b, c)  # (b, c)
        y = self.fc(y).view(b, 1, c)  # (b, 1, c)

        return x * y.expand_as(x)

class WtNet(nn.Module):
    def __init__(self, wavelet="db4", level=4, axis=2, in_seq=7, hid_seq=512, out_seq=7,in_chn=7,
                 activ="gelu", drop=0.0, pred_len=96, filter_len=8, use_channel_attn=True,attn_type = "efficient"):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.in_seq = in_seq
        self.hid_seq = hid_seq
        self.out_seq = out_seq
        self.in_chn =in_chn
        self.activ = activ
        self.drop = drop
        self.axis = axis
        self.use_channel_attn = use_channel_attn
        self.mlp_block_list = nn.ModuleList()
        self.feature = []
        self.seq_len = pred_len * 2
        # Calculate feature lengths
        for i in range(level):
            self.seq_len = (self.seq_len + filter_len - 1) // 2
            self.feature.append(self.seq_len)
            if i == level - 1:
                self.feature.append(self.seq_len)
        self.feature.reverse()

        # Initialize channel attention if needed
        if self.use_channel_attn:
            self.channel_attentions = nn.ModuleList()
            for _ in range(self.level + 1):
                if attn_type == 'efficient':
                    self.channel_attentions.append(EfficientChannelAttention(in_chn))
                else:
                    self.channel_attentions.append(ChannelAttention(in_chn))

        self.feature_mlp = MLPBlock(1, self.out_seq, self.hid_seq, self.out_seq, self.activ, self.drop)

        for i in range(self.level + 1):
            self.mlp_block_list.append(MLPBlock(1, self.feature[i], hid_seq, out_seq, activ, drop))

    def forward(self, x, x_time):
        # x : b l c
        p_head_cpu = x.cpu().detach().numpy()
        coeffs = pywt.wavedec(p_head_cpu, wavelet=self.wavelet, level=self.level, axis=self.axis)
        coeffs_list = [torch.from_numpy(c).to(x.device) for c in coeffs]

        pred_list = []
        for i, mlp_block in enumerate(self.mlp_block_list):
            # pred = mlp_block(coeffs_list[i]).float()
            #
            # # Apply channel attention if enabled
            # if self.use_channel_attn:
            #     pred = self.channel_attentions[i](pred)
            pred = coeffs_list[i]
            if self.use_channel_attn:
                pred = self.channel_attentions[i](pred) + pred
            pred = mlp_block(pred).float()

            pred_list.append(pred)

        output = reduce(torch.stack(pred_list, 0), "h b l c -> b l c", "sum")
        y_pred = self.feature_mlp(output)
        return y_pred, x
'''
