import os

import pandas as pd
import torch.utils.data as data
from torch.utils.data import DataLoader
import random


class DataSet(data.Dataset):
    files: list = list()
    epochs: int = 0
    appliances: list = list()
    status: list = list()

    def __init__(self, path_data: str, config: dict, subset: str = "train"):
        self.power_scale = config["power_scale"]
        self.border = config["border"]
        self.length = config["input_len"]
        self.threshold = config["threshold"]
        self._list_files(path_data, config, subset)
        self._get_parameters_from_file()

    @staticmethod
    def _open_file(path_file: str) -> pd.DataFrame:
        df = pd.read_csv(path_file, index_col=0)
        return df

    def _list_files(self, path_data: str, config: dict, subset: str):
        files = []
        for building in sorted(os.listdir(path_data)):
            path_building = os.path.join(path_data, building)
            dataset, house = building.rsplit("_", 1)
            files_of_dataset = sorted(
                [
                    os.path.join(path_building, file)
                    for file in os.listdir(path_building)
                ]
            )
            # Shuffle if requested
            if config.get("random", False):
                random.seed(config.get("random_seed", 0))
                random.shuffle(files_of_dataset)
            if int(house) in config[subset]["buildings"][dataset]:
                val_idx = int(len(files_of_dataset) * config["train_size"])
                test_idx = val_idx + int(len(files_of_dataset) * config["valid_size"])
                if subset == "train":
                    files += files_of_dataset[:val_idx]
                elif subset == "validation":
                    files += files_of_dataset[val_idx:test_idx]
                elif subset == "test":
                    files += files_of_dataset[test_idx:]
                else:
                    raise KeyError(f"Subset not found: {subset}")
        self.files = files
        self.epochs = len(files)
        print(f"{self.epochs} data points found for {subset}")

    def _get_parameters_from_file(self):
        df = self._open_file(self.files[0])
        appliances = [t for t in df.columns if not t.endswith("_status")]
        appliances.remove("aggregate")
        self.appliances = appliances
        self.status = [t + "_status" for t in appliances]
        self.length = df.shape[0]
        self._idx_start = self.border
        self._idx_end = self.length - self.border

    def __getitem__(self, index):
        path_file = self.files[index]
        df = self._open_file(path_file)
        x = df["aggregate"].values / self.power_scale
        y = (
            df[self.appliances].iloc[self._idx_start : self._idx_end].values
            / self.power_scale
        )
        s = df[self.status].iloc[self._idx_start : self._idx_end].values
        return x, y, s

    def __len__(self):
        return self.epochs


def return_dataloader(
    path_data: str,
    config_data: dict,
    config_model: dict,
    subset: str = "train",
    shuffle: bool = True,
):
    dataset = DataSet(path_data, config_data, subset=subset)
    dataloader = DataLoader(
        dataset=dataset, batch_size=config_model["batch_size"], shuffle=shuffle
    )
    return dataloader
