import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft
import matplotlib.pyplot as plt
from timm.models.layers import DropPath

class FeelNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, Filter=True,
                 dropout_rate=0.5, kern_length=5, F1=8, D=2, F2=16, drop=0.):    #kern_length=64, F1=8, D=2, F2=16
        super(FeelNet, self).__init__()

        self.drop = nn.Dropout(drop)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2)),
            nn.BatchNorm2d(F1),  # [B, F1, C, T]

            Conv2dWithConstraint(F1, F1 * D, (Chans, 1), groups=F1, padding=0, bias=False, max_norm=1),

            nn.BatchNorm2d(F1 * D),  # [B, F1*D, 1, T]
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),  # [B, F1*D, C, T->T/4]
            nn.Dropout(dropout_rate)
        )

        self.conv2 = nn.Sequential(
            # nn.Conv2d(F1 * D, F2, (1, 64), bias=False, groups=F1 * D, padding=(0, 64 // 2)),  #[B, F2, Depth, T]
            # nn.Conv2d(F1 * D, F2, (1, 32), bias=False, groups=F1 * D, padding=(0, 32 // 2)),  #[B, F2, Depth, T]
            # nn.Conv2d(F1 * D, F2, (1, 16), bias=False, groups=F1 * D, padding=(0, 16 // 2)),  #[B, F2, Depth, T]
            # nn.Conv2d(F1 * D, F2, (1, 1), padding=0, bias=False, stride=1),  # [B, F2, Depth, T]

            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),  # [B, F2, Depth, T->T/8]
            nn.Dropout(dropout_rate)
        )

        self.parallel_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(F1 * D, F2, (1, 16), groups=F1, padding=(0, 16 // 2), bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), padding=0, bias=False, stride=1),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8), stride=(1, 8)),  # [B, F2, Depth, T->T/8]
                nn.Dropout(dropout_rate)
            ),
            nn.Sequential(
                nn.Conv2d(F1 * D, F2, (1, 32), groups=F1, padding=(0, 32 // 2), bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), padding=0, bias=False, stride=1),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8), stride=(1, 8)),  # [B, F2, Depth, T->T/8]
                nn.Dropout(dropout_rate)
            ),
            nn.Sequential(
                nn.Conv2d(F1 * D, F2, (1, 64), groups=F1, padding=(0, 64 // 2), bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), padding=0, bias=False, stride=1),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8), stride=(1, 8)),  # [B, F2, Depth, T->T/8]
                nn.Dropout(dropout_rate)
            )

        ])

        self.position_encoding = PositionalEncoding(Samples, Samples // 4)  # F2
        # self.fft_block = RhythmFilterBlock(N=Samples, dim=Chans, filter_enabled=Filter)
        self.fft_block = nn.Sequential(RhythmFilterlayer(Chans=Chans, Samples=Samples, filter_enabled=Filter)
                                       )

        self.norm1 = nn.LayerNorm(Samples)
        self.norm2 = nn.LayerNorm(Samples)
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(F2, nb_classes)

    def forward(self, x):

        # print(x.min(), x.max(), x.mean(), x.var())
        # x = x - x.mean(dim=-1, keepdims=True)
        # x = x + torch.normal(mean=0, std=60.0, size=x.shape, device=x.device)  # 0 1.0 10.0 20.0 40 60

        x1 = x.squeeze(1)  # Remove the temporal dimension

        # FFT Block
        FFT = self.position_encoding(x1)

        FFT = self.fft_block(FFT)
        # Depthwise and Separable ConvBlock
        x2 = self.norm2(FFT)
        x2 = x2.unsqueeze(1)

        Conv1 = self.conv1(x2)
        out = [conv(Conv1) for conv in self.parallel_conv]
        Conv2 = torch.cat(out, dim=-1)
        # Conv2 = self.conv2(Conv1)

        Conv = Conv2.squeeze(2)

        x = self.global_avg_pooling(Conv)
        x = x.squeeze(2)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term2 = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        div_term1 = torch.exp(torch.arange(1, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term2)
        pe[:, 1::2] = torch.cos(position * div_term1)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, x.size(1), :]
        return x

# GFNet(NeurIPS 2021)/ AFNO(ICLR 2022)/ TSLANet(ICML 2024)
# class RhythmFilterBlock(nn.Module):
#     def __init__(self, dim, filter_enabled=True):
#         super().__init__()
#         self.complex_weight_freq = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
#         self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
#         self.filter_enabled = filter_enabled
#         self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
#
#         nn.init.trunc_normal_(self.complex_weight_freq, std=.02)
#         nn.init.trunc_normal_(self.complex_weight, std=.02)
#
#         self.learnable_param1 = nn.Parameter(torch.tensor(1.0))
#         self.learnable_param2 = nn.Parameter(torch.tensor(0.5))
#
#         # Define frequency band ranges
#         self.delta_band = (0.5, 4)
#         self.theta_band = (4, 8)
#         self.alpha_band = (8, 13)
#         self.beta_band = (13, 30)
#         self.gamma_band = (30, 60)
#
#     def get_freq_mask(self, freqs):
#         delta_mask = (freqs >= self.delta_band[0]) & (freqs <= self.delta_band[1])
#         theta_mask = (freqs >= self.theta_band[0]) & (freqs <= self.theta_band[1])
#         alpha_mask = (freqs >= self.alpha_band[0]) & (freqs <= self.alpha_band[1])
#         beta_mask = (freqs >= self.beta_band[0]) & (freqs <= self.beta_band[1])
#         gamma_mask = (freqs >= self.gamma_band[0]) & (freqs <= self.gamma_band[1])
#
#         freq_mask = torch.stack([delta_mask, theta_mask, alpha_mask, beta_mask, gamma_mask], dim=0).float()
#         return freq_mask
#
#     def calculate_psd(self, x_fft, freq_mask):
#         psd_values = []
#         for i in range(5):
#             masked_fft = x_fft * freq_mask[i].unsqueeze(0).unsqueeze(-1)
#             psd = torch.abs(masked_fft).pow(2).sum(dim=-1)
#             psd_values.append(psd)
#         return torch.stack(psd_values)
#
#     def calculate_thresholds(self, psd_values):
#         thresholds = []
#         for i in range(5):
#             median_psd = psd_values[i].median(dim=1, keepdim=True).values
#             epsilon = 1e-6  # Small constant to avoid division by zero
#             normalized_energy = psd_values[i] / (median_psd + epsilon)
#             threshold = torch.quantile(normalized_energy, self.threshold_param)
#             thresholds.append(threshold)
#         return torch.stack(thresholds)
#
#     def apply_thresholds(self, x_fft, freq_mask, thresholds, psd_values):
#         filtered_signals = []
#         for i in range(5):
#             masked_fft = x_fft * freq_mask[i].unsqueeze(0).unsqueeze(-1)
#             psd = psd_values[i]
#             mask = psd > thresholds[i]
#
#             filtered_signal = masked_fft * mask.unsqueeze(-1)
#             weight_freq = torch.view_as_complex(self.complex_weight_freq)
#             x_filtered = filtered_signal * weight_freq
#             filtered_signals.append(x_filtered)
#         return torch.stack(filtered_signals, dim=1).sum(dim=1)
#
#     def forward(self, x_in):
#         B, N, C = x_in.shape
#         dtype = x_in.dtype
#         x = x_in.to(torch.float32)
#
#         # Plot input signal spectrum
#         #self.plot_spectrum(x, N, 'Input Signal Spectrum')
#
#         # FFT
#         x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
#         #equency_list = abs(x_fft).mean(0).mean(-1)
#         #print(frequency_list)
#         Global_weight = torch.view_as_complex(self.complex_weight)
#         x_weighted = x_fft * Global_weight
#
#         # Plot weighted signal spectrum
#         #self.plot_spectrum(torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho'), N, 'Weighted Signal Spectrum')
#
#         if self.filter_enabled:
#             freqs = torch.fft.rfftfreq(N, 1.0/128).to(x.device)
#             freq_mask = self.get_freq_mask(freqs)
#             psd_values = self.calculate_psd(x_fft, freq_mask)
#
#             thresholds = self.calculate_thresholds(psd_values)
#
#             x_filtered = self.apply_thresholds(x_fft, freq_mask, thresholds, psd_values)
#             x_weighted += x_filtered
#
#             # Plot filtered signal spectrum
#             # self.plot_spectrum(torch.fft.irfft(x_filtered, n=N, dim=1, norm='ortho'), N, 'Filtered Signal Spectrum')
#
#         # Apply additional learnable parameters dynamically
#         #x_weighted = x_weighted * self.learnable_param1 + x_filtered * self.learnable_param2
#         x_weighted = x_weighted * self.learnable_param1 + self.learnable_param2
#
#         # IFFT
#         x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
#
#         # Plot output signal spectrum
#         #self.plot_spectrum(x, N, 'Output Signal Spectrum')
#
#         x = x.to(dtype)
#         x = x.view(B, N, C)
#
#         return x
class RhythmFilterlayer(nn.Module):
    def __init__(self, Samples, Chans, filter_enabled=True, drop_path=0.):
        super(RhythmFilterlayer, self).__init__()
        self.fft_block = RhythmFilterBlock(N=Samples, dim=Chans, filter_enabled=filter_enabled)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm1 = nn.LayerNorm(Samples)

    def forward(self, x):
        # FFT = self.norm1(x)  # [B, N, C]
        FFT = x.permute(0, 2, 1)  # Swap dimensions for transformer input
        FFT = self.fft_block(FFT)
        FFT = FFT.permute(0, 2, 1)  # Swap dimensions back
        FFT = self.drop_path(FFT)
        FFT = FFT + x

        # FFT = self.norm2(FFT)  # 避免过拟合

        return FFT

class RhythmFilterBlock(nn.Module):
    def  __init__(self, N, dim, filter_enabled=True):
        super().__init__()

        self.complex_weight_freq = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.filter_enabled = filter_enabled

        self.shared_threshold = False
        if self.shared_threshold:
            self.threshold_param = nn.Parameter(torch.rand(1) * 0.5, requires_grad=True)  ###
        else:
            self.threshold_param = nn.Parameter(torch.rand(5) * 0.5, requires_grad=True)  # 5

        nn.init.trunc_normal_(self.complex_weight_freq, std=.02)
        nn.init.trunc_normal_(self.complex_weight, std=.02)

        self.learnable_paramµ = nn.Parameter(torch.tensor(0.5))  ###
        self.learnable_paramρ = nn.Parameter(torch.tensor(1.0))

        # Define frequency band ranges
        self.bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 50)
        }

        if self.filter_enabled:
            band_masks = []
            freqs = torch.fft.rfftfreq(N, 1.0/128).cuda()
            for i, (name, (low, high)) in enumerate(self.bands.items()):
                band_mask = (freqs >= low) & (freqs <= high)
                band_masks.append(band_mask)
            self.freq_mask = torch.stack(band_masks, dim=0).float()

    def calculate_psd(self, x_fft, freq_mask):
        psd_values = []
        for i in range(5):
            masked_fft = x_fft * freq_mask[i].unsqueeze(0).unsqueeze(-1)
            psd = torch.abs(masked_fft).pow(2)
            psd_values.append(psd)
        return torch.stack(psd_values)

    def calculate_thresholds(self, psd_values):
        thresholds = []
        for i in range(5):
            if self.shared_threshold:
                threshold = self.threshold_param
            else:
                threshold = self.threshold_param[i]
            thresholds.append(threshold)
        # print(thresholds)
        return torch.stack(thresholds)

    def apply_thresholds(self, x_fft, freq_mask, thresholds, psd_values):
        # 使用全局池化获取每个输入样本的全局特征
        # global_features = x_fft.mean(dim=(-1, -2), keepdim=True)  # 假设 x 的形状为 [B, N, C]
        # device = global_features.device
        # # 使用一个简单的线性变换生成权重
        # linear_layer = nn.Linear(global_features.shape[-1], 5).to(device)
        # global_features = global_features.real
        # weights_real = torch.sigmoid(linear_layer(global_features)).to(device)

        # weights = torch.complex(weights_real, torch.zeros_like(weights_real))  # 输出 [B, 5]
        # weights = weights.view(-1, 5, 1, 1)  # 调整为 [B, 5, 1, 1] 以匹配后续运算

        # 初始化结果张量
        # weighted_sum_filtered_signals = torch.zeros_like(x_fft)

        filtered_signals = []
        for i in range(5):
            masked_fft = x_fft * freq_mask[i].unsqueeze(0).unsqueeze(-1)

            # median_psd = psd_values[i].mean(dim=1, keepdim=True)
            median_psd = psd_values[i].median(dim=1, keepdim=True).values
            epsilon = 1e-6  # Small constant to avoid division by zero
            normalized_energy = psd_values[i] / (median_psd + epsilon)
            # mask = psd > thresholds[i].unsqueeze(0).unsqueeze(-1)
            mask = ((normalized_energy > thresholds[i]).float() - thresholds[i]).detach() + thresholds[i]

            filtered_signal = masked_fft * mask
            weight_freq = torch.view_as_complex(self.complex_weight_freq)

            x_filtered = filtered_signal * weight_freq

            filtered_signals.append(x_filtered)

            # 使用动态权重
            # weighted_sum_filtered_signals += weights[:, i, :, :] * x_filtered

        return torch.stack(filtered_signals, dim=1).sum(dim=1)
        # return weighted_sum_filtered_signals

    def forward(self, x_in):
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # FFT
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.filter_enabled:
            psd_values = self.calculate_psd(x_fft, self.freq_mask)
            thresholds = self.calculate_thresholds(psd_values)
            x_filtered = self.apply_thresholds(x_fft, self.freq_mask, thresholds, psd_values)
            # x_weighted += x_filtered

            # Apply additional learnable parameters dynamically
            x_weighted = x_weighted * self.learnable_paramµ + x_filtered * self.learnable_paramρ   #self.learnable_param2
            # x_weighted = x_weighted * self.learnable_param1 + self.learnable_param2 * 0

        # IFFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  #x1

        # # 创建子图
        # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        #
        # # 对选择的 batch 和 channel 进行傅里叶变换
        # frequency_data = torch.fft.fft(x[1, 2, :])  # 一维数据
        # amplitude = torch.abs(frequency_data).cpu().numpy()  # 获取幅度谱并转换为 NumPy
        #
        # # 绘制频谱图
        # frequencies = torch.fft.fftfreq(len(amplitude), d=1 / 128).cpu().numpy()  # 频率轴
        # axs[0].plot(frequencies[:len(frequencies) // 2], amplitude[:len(amplitude) // 2])
        # axs[0].set_title('Before Filter')
        # axs[0].set_xlabel('Frequency (Hz)')
        # axs[0].set_ylabel('Amplitude')
        # axs[0].grid(True)
        #
        # frequency_data_1 = torch.fft.fft(x1[1, 2, :])  # 一维数据
        # amplitude_2 = torch.abs(frequency_data_1).detach().cpu().numpy()  # 获取幅度谱并转换为 NumPy
        #
        # # 绘制滤波后的频域特征图
        # frequencies_2 = torch.fft.fftfreq(len(amplitude_2), d=1 / 128).cpu().numpy()  # 频率轴
        # axs[1].plot(frequencies_2[:len(frequencies_2) // 2], amplitude_2[:len(amplitude_2) // 2]*10)
        #
        # # 添加不同频段的背景颜色
        # # 第一个频段 0.5-4Hz，颜色为浅红色
        # axs[1].axvspan(0.5, 4, facecolor='red', alpha=0.3)
        #
        # # 第二个频段 4-8Hz，颜色为浅绿色
        # axs[1].axvspan(4, 8, facecolor='green', alpha=0.3)
        #
        # # 第三个频段 8-14Hz，颜色为浅蓝色
        # axs[1].axvspan(8, 14, facecolor='blue', alpha=0.3)
        #
        # # 第四个频段 14-30Hz，颜色为浅黄色
        # axs[1].axvspan(14, 30, facecolor='yellow', alpha=0.3)
        #
        # # 第五个频段 30-60Hz，颜色为浅灰色
        # axs[1].axvspan(30, 60, facecolor='gray', alpha=0.3)
        #
        # # 标注每个频段
        # axs[1].text(2, 0.5, 'delta', fontsize=12, color='red')
        # axs[1].text(6, 0.4, 'theta', fontsize=12, color='green')
        # axs[1].text(11, 0.5, 'alpha', fontsize=12, color='blue')
        # axs[1].text(22, 0.4, 'beta', fontsize=12, color='orange')
        # axs[1].text(45, 0.5, 'gamma', fontsize=12, color='gray')
        # axs[1].set_title('After Filter')
        # axs[1].set_xlabel('Frequency (Hz)')
        # axs[1].set_ylabel('Amplitude')
        # axs[1].grid(True)
        #
        # # 显示图像
        # plt.tight_layout()
        # plt.show()

        return x

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

