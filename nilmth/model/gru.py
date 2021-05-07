import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nilmth.model.pytorch import TorchModel


class Dense(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class BiGRU(nn.Module):
    def __init__(self, input_len=510, output_len=480, out_channels=1, dropout=0.1):
        super(BiGRU, self).__init__()

        padding = 2
        kernel_size = int((input_len - output_len + 4 * padding + 2) / 2)

        self.drop = nn.Dropout(dropout)

        self.conv1 = nn.Conv1d(1, 16, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=kernel_size, padding=padding)

        self.gru1 = nn.GRU(8, 64, batch_first=True, bidirectional=True, dropout=dropout)
        self.gru2 = nn.GRU(128, 128, batch_first=True, bidirectional=True)

        self.dense = Dense(256, 64)
        self.regressor = nn.Conv1d(64, out_channels, kernel_size=1, padding=0)
        self.activation = nn.Conv1d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.drop(conv1))

        gru1 = self.gru1(self.drop(conv2.permute(0, 2, 1)))[0]
        gru2 = self.gru2(self.drop(gru1))[0]

        dense = self.dense(self.drop(gru2))

        power = self.regressor(self.drop(dense.permute(0, 2, 1)))
        status = self.activation(self.drop(F.relu(dense.permute(0, 2, 1))))

        return power, status


class GRUModel(TorchModel):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

        msg = "Difference between input and output lens must be even"
        assert (self.input_len - self.output_len) % 2 == 0, msg

        self.model = BiGRU(
            input_len=self.input_len,
            output_len=self.output_len,
            out_channels=len(self.appliances),
            dropout=self.dropout,
        ).to(device=self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.pow_criterion = nn.MSELoss()
        self.act_criterion = nn.BCEWithLogitsLoss()
