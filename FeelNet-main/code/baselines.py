
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# %% Deep convnet - Baseline 1
from torch.nn.utils import weight_norm


class deepConvNet(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernalSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs),
            Conv2dWithConstraint(25, 25, (nChan, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(inF, outF, kernalSize, max_norm=0.5, *args, **kwargs))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass=2, dropoutP=0.25, *args, **kwargs):
        super(deepConvNet, self).__init__()

        kernalSize = (1, 5)   # Please note that the kernel size in the origianl paper is (1, 10), we found when the segment length is shorter than 4s (1s, 2s, 3s) larger kernel size will 
                              # cause network error. Besides using (1, 5) when EEG segment is 4s gives slightly higher ACC and F1 with a smaller model size. 
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]

        firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, nChan)
        middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, kernalSize)
                                       for inF, outF in zip(nFiltLaterLayer, nFiltLaterLayer[1:])])

        self.allButLastLayers = nn.Sequential(firstLayer, middleLayers)

        self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))

    def forward(self, x):
        x = self.allButLastLayers(x)
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x

# %% Shallow convnet - Baseline 3
class shallowConvNet(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernalSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs),
            Conv2dWithConstraint(40, 40, (nChan, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(outF),
        )

    def calculateOutSize(self, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, nChan, nTime)
        block_one = self.firstLayer
        avg = self.avgpool
        dp = self.dp
        out = torch.log(block_one(data).pow(2))
        out = avg(out)
        out = dp(out)
        out = out.view(out.size()[0], -1)
        return out.size()

    def __init__(self, nChan, nTime, nClass=2, dropoutP=0.25, *args, **kwargs):
        super(shallowConvNet, self).__init__()

        kernalSize = (1, 25)
        nFilt_FirstLayer = 40

        self.firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, nChan)
        self.avgpool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dp = nn.Dropout(p=dropoutP)
        self.fSize = self.calculateOutSize(nChan, nTime)
        self.lastLayer = nn.Linear(self.fSize[-1], nClass)

    def forward(self, x):
        x = self.firstLayer(x)
        x = torch.log(self.avgpool(x.pow(2)))
        x = self.dp(x)
        x = x.view(x.size()[0], -1)
        x = self.lastLayer(x)

        return x


# %% EEGNet Baseline 2
class eegNet(nn.Module):
    def initialBlocks(self, dropoutP, *args, **kwargs):
        block1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, self.C1),
                          padding=(0, self.C1 // 2), bias=False),
                nn.BatchNorm2d(self.F1),


                # DepthwiseConv2D ====================================================
                Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
                                     padding=0, bias=False, max_norm=1,
                                     groups=self.F1),
                # ====================================================

                nn.BatchNorm2d(self.F1 * self.D),
                nn.ELU(),
                nn.AvgPool2d((1, 4), stride=4),
                nn.Dropout(p=dropoutP))
        block2 = nn.Sequential(
                # SeparableConv2D ====================================================
                nn.Conv2d(self.F1 * self.D, self.F1 * self.D,  (1, 22),
                                     padding=(0, 22//2), bias=False,
                                     groups=self.F1 * self.D),
                nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                          stride=1, bias=False, padding=0),
                # ====================================================

                nn.BatchNorm2d(self.F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8), stride=8),
                nn.Dropout(p=dropoutP)
                )
        return nn.Sequential(block1, block2)

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
                nn.Conv2d(inF, outF, kernalSize, *args, **kwargs))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass=2,
                 dropoutP=0.25, F1=8, D=2,
                 C1=64, *args, **kwargs):
        super(eegNet, self).__init__()
        self.F2 = D*F1
        self.F1 = F1
        self.D = D
        self.nTime = nTime
        self.nClass = nClass
        self.nChan = nChan
        self.C1 = C1

        self.firstBlocks = self.initialBlocks(dropoutP)
        self.fSize = self.calculateOutSize(self.firstBlocks, nChan, nTime)
        self.lastLayer = self.lastBlock(self.F2, nClass, (1, self.fSize[1]))

    def forward(self, x):
        x = self.firstBlocks(x)
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x

# %%EEG-TCNet  - Baseline 5
#
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)

        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()


    def forward(self, x):
        # 先裁剪 后加入下采样，维度先降维后升维
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# 因果卷积
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2) -> None:
        super(TCNNet, self).__init__()

        self.tcn_block = TemporalConvNet(num_inputs, num_channels, kernel_size)
        # self.tcn_block =  TemporalConvNet(num_inputs=self.F2,num_channels=[tcn_filters,tcn_filters],kernel_size=tcn_kernelSize)

    def forward(self, x):
        if len(x.shape) == 4:
            data = torch.rand(x.shape)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data = data.to(device)

            for i in range(x.shape[2]):
                data[:, :, i, :] = self.tcn_block(x[:, :, i, :])
            x = data
        else:
            #x = torch.squeeze(x, dim=2)
            x = self.tcn_block(x)

        return x


class EEGTCNet(nn.Module):
    def __init__(self,  nChan, nTime, nClass=2, dropoutP=0.25, tcn_filters=64, tcn_dropout=0.3, tcn_kernelSize=4, F1=8, D=2, C1=64, *args):
        super(EEGTCNet, self).__init__()
        self.nChan = nChan
        self.nTime = nTime
        Class = nClass
        dropoutP = dropoutP
        self.F2 = D*F1
        self.F1 = F1
        self.D = D
        self.C1 = C1

        self.eegnet = self.initialBlocks(dropoutP)
        self.flatten = nn.Flatten()
        self.tcn_block = TCNNet(self.F2, [tcn_filters, tcn_filters], tcn_kernelSize, tcn_dropout)
        self.linear = LinearWithConstraint(in_features=tcn_filters, out_features=Class, max_norm=.25)
        self.softmax = nn.Softmax(dim=-1)

    def initialBlocks(self, dropoutP, *args, **kwargs):
        block1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, self.C1),
                          padding=(0, self.C1 // 2), bias=False),
                nn.BatchNorm2d(self.F1),


                # DepthwiseConv2D ====================================================
                Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
                                     padding=0, bias=False, max_norm=1,
                                     groups=self.F1),
                # ====================================================

                nn.BatchNorm2d(self.F1 * self.D),
                nn.ELU(),
                nn.AvgPool2d((1, 4), stride=4),
                nn.Dropout(p=dropoutP))
        block2 = nn.Sequential(
                # SeparableConv2D ====================================================
                nn.Conv2d(self.F1 * self.D, self.F1 * self.D,  (1, 32),
                                     padding=(0, 32//2), bias=False,
                                     groups=self.F1 * self.D),
                nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                          stride=1, bias=False, padding=0),
                # ====================================================

                nn.BatchNorm2d(self.F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8), stride=8),
                nn.Dropout(p=dropoutP)
                )
        return nn.Sequential(block1, block2)

    def forward(self, x):
        # EEGNet
        x = self.eegnet(x)
        # 先进行降维处理，然后进入TCN
        x = torch.squeeze(x, dim=2)
        x = self.tcn_block(x)
        x = x[:, :, -1]

        x = self.flatten(x)
        x = self.linear(x)
        out = self.softmax(x)

        return out

    # %%ERTNet - Baseline 4
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.ELU()
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        attn_output, _ = self.att(inputs, inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output)
        out = self.layernorm2(out1 + ffn_output)
        return out

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

class ERTNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128,
                 dropout_rate=0.5, kern_length=64, F1=8, heads=8,
                 D=2, F2=16):
        super(ERTNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2)),
            nn.BatchNorm2d(F1),
            Conv2dWithConstraint(F1, F1 * D, (Chans, 1), groups=F1, padding=0, bias=False, max_norm=1),

            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(dropout_rate)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 16 // 2), bias=False, groups=F1 * D),
            nn.Conv2d(F1 * D, F2, (1, 1), padding=0, bias=False, stride=1
                      ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(dropout_rate)
        )
        self.position_encoding = PositionalEncoding(F2, Samples // 4)
        self.transformer_block = TransformerBlock(embed_dim=F2, num_heads=heads, dropout_rate=dropout_rate)
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(F2, nb_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.squeeze(2)  # Remove the temporal dimension

        x = self.position_encoding(x)
        x = x.permute(0, 2, 1)  # Swap dimensions for transformer input
        x = self.transformer_block(x)
        x = x.permute(0, 2, 1)  # Swap dimensions back
        x = self.global_avg_pooling(x)
        x = x.squeeze(2)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)

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


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)
