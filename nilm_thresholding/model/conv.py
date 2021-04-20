import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nilm_thresholding.model.pytorch import TorchModel

"""
Source: https://github.com/lmssdd/TPNILM
Check the paper
Non-Intrusive Load Disaggregation by Convolutional
Neural Network and Multilabel Classification
by Luca Massidda, Marino Marrocu and Simone Manca
"""


class _Encoder(nn.Module):
    def __init__(
        self,
        in_features=3,
        out_features=1,
        kernel_size=3,
        padding=1,
        stride=1,
        dropout=0.1,
    ):
        super(_Encoder, self).__init__()
        self.conv = nn.Conv1d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.bn(F.relu(self.conv(x))))


class _TemporalPooling(nn.Module):
    def __init__(self, in_features=3, out_features=1, kernel_size=2, dropout=0.1):
        super(_TemporalPooling, self).__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.kernel_size)
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm1d(out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(F.relu(x))
        x = self.drop(
            F.interpolate(
                x, scale_factor=self.kernel_size, mode="linear", align_corners=True
            )
        )
        return x


class _Decoder(nn.Module):
    def __init__(self, in_features=3, out_features=1, kernel_size=2, stride=2):
        super(_Decoder, self).__init__()
        self.conv = nn.ConvTranspose1d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        return self.conv(x)


class _PTPNet(nn.Module):
    def __init__(self, output_len=480, out_channels=1, init_features=32, dropout=0.1):
        super(_PTPNet, self).__init__()

        p = 2
        k = 1
        features = init_features
        self.encoder1 = _Encoder(1, features, kernel_size=3, padding=0, dropout=dropout)
        # (batch, input_len - 2, 32)
        self.pool1 = nn.MaxPool1d(kernel_size=p, stride=p)

        self.encoder2 = _Encoder(
            features * 1 ** k,
            features * 2 ** k,
            kernel_size=3,
            padding=0,
            dropout=dropout,
        )
        # (batch, [input_len - 6] / 2, 64)
        self.pool2 = nn.MaxPool1d(kernel_size=p, stride=p)

        self.encoder3 = _Encoder(
            features * 2 ** k,
            features * 4 ** k,
            kernel_size=3,
            padding=0,
            dropout=dropout,
        )
        # (batch, [input_len - 12] / 4, 128)
        self.pool3 = nn.MaxPool1d(kernel_size=p, stride=p)

        self.encoder4 = _Encoder(
            features * 4 ** k,
            features * 8 ** k,
            kernel_size=3,
            padding=0,
            dropout=dropout,
        )
        # (batch, [input_len - 30] / 8, 256)

        # Compute the output size of the encoder4 layer
        # (batch, S, 256)
        s = output_len / 8

        self.tpool1 = _TemporalPooling(
            features * 8 ** k,
            features * 2 ** k,
            kernel_size=int(s / 12),
            dropout=dropout,
        )
        self.tpool2 = _TemporalPooling(
            features * 8 ** k,
            features * 2 ** k,
            kernel_size=int(s / 6),
            dropout=dropout,
        )
        self.tpool3 = _TemporalPooling(
            features * 8 ** k,
            features * 2 ** k,
            kernel_size=int(s / 3),
            dropout=dropout,
        )
        self.tpool4 = _TemporalPooling(
            features * 8 ** k,
            features * 2 ** k,
            kernel_size=int(s / 2),
            dropout=dropout,
        )

        self.decoder = _Decoder(
            2 * features * 8 ** k, features * 1 ** k, kernel_size=p ** 3, stride=p ** 3
        )

        self.activation = nn.Conv1d(
            features * 1 ** k, out_channels, kernel_size=1, padding=0
        )

        self.power = nn.Conv1d(
            features * 1 ** k, out_channels, kernel_size=1, padding=0
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        tp1 = self.tpool1(enc4)
        tp2 = self.tpool2(enc4)
        tp3 = self.tpool3(enc4)
        tp4 = self.tpool4(enc4)

        dec = self.decoder(torch.cat([enc4, tp1, tp2, tp3, tp4], dim=1))

        pw = self.power(dec)
        act = self.activation(F.relu(dec))

        return pw, act


class ConvModel(TorchModel):
    def __init__(self, config: dict):
        super().__init__(config)

        # The time series will undergo three convolutions + poolings
        # This will give a series of size (batch, S, 256)
        # Where S = output_len / 8
        # That series will them pass by four different filters
        # The output of the four filters must have the same size
        # For this reason, S must be a multiple of 12
        if self.output_len % 96 != 0:
            s = round(self.output_len / 96)
            raise ValueError(
                f"output_len {self.output_len} is not valid.\nClosest"
                f" valid value is {96 * s}"
            )

        self.model = _PTPNet(
            output_len=self.output_len,
            out_channels=len(self.appliances),
            init_features=self.init_features,
            dropout=self.dropout,
        ).to(device=self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
