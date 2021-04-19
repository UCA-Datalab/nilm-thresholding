import numpy as np
import torch.utils.data as data
import os
import itertools


class Power(data.Dataset):
    def __init__(
        self,
        meter=None,
        appliance=None,
        status=None,
        length=512,
        border=16,
        power_scale=2000.0,
        train=False,
    ):
        self.length = length
        self.border = border
        self.power_scale = power_scale
        self.train = train

        self.meter = meter.copy() / self.power_scale
        self.appliance = appliance.copy() / self.power_scale
        self.status = status.copy()

        self.epochs = (len(self.meter) - 2 * self.border) // self.length

    def __getitem__(self, index):
        i = index * self.length + self.border
        if self.train:
            i = np.random.randint(
                self.border, len(self.meter) - self.length - self.border
            )

        x = self.meter.iloc[
            i - self.border : i + self.length + self.border
        ].values.astype("float32")
        y = self.appliance.iloc[i : i + self.length].values.astype("float32")
        s = self.status.iloc[i : i + self.length].values.astype("float32")
        x -= x.mean()

        return x, y, s

    def __len__(self):
        return self.epochs


class DataLoader:
    def __init__(
        self,
        path_data: str,
        buildings: dict = None,
        batch_size: int = 32,
        power_scale: float = 2000,
        border: int = 16,
    ):
        self.batch_size = batch_size
        self.power_scale = power_scale
        self.border = border

        folders = [k + "_" + str(i) for k, v in buildings.items() for i in v]
        folders = [
            os.path.join(path_data, f) for f in folders if f in os.listdir(path_data)
        ]
        self.files = [os.path.join(f, x) for f in folders for x in os.listdir(f)]
        print(len(self.files))
