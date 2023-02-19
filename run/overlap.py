import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from nilmth.data.dataloader import DataLoader
from nilmth.utils.config import load_config

LIST_CONFIGS = [
    "./configs/ConvModel_mp_classw_1.toml",
    "./configs/ConvModel_vs_classw_1.toml",
    "./configs/ConvModel_at_classw_1.toml",
]


def main(
    path_configs: Optional[str] = None,
    path_data: str = "./data-prep",
    path_out: str = "overlap.csv",
):
    """
    Reads all available config files from `path_configs` and generates
    a csv with one row per configuration, stating the percentage of overlapping
    ON activations across the whole dataset, for that specific threshold
    configuration

    WARNING: config files should be generated first with
    `python nilmth/generate_config_files.py`

    Parameters
    ----------
    path_configs : str, optional
        Path to the folder with config files, by default None
    path_data : str, optional
        Path to the processed data, by default "./data-prep"
    path_out : str, optional
        Path where the csv is stored, by default "overlap.csv"
    """
    if path_configs is None:
        list_configs = [Path(s) for s in LIST_CONFIGS]
    else:
        path_configs = Path(path_configs)
        list_configs = list(path_configs.iterdir())
    # Initialize the list that will contain the overlaps for each configuration
    list_counts = []
    # Temporal threshold file
    path_threshold = "threshold_overlap.toml"
    # Loop over all possible configurations
    for path_config in list_configs:
        config = load_config(path_config, "model")
        # Loop over both subsets
        for subset in ["train", "test"]:
            loader = DataLoader(
                path_data, subset=subset, path_threshold=path_threshold, **config
            )

            ser = np.stack(
                [
                    loader.get_appliance_series(app, "status")
                    for app in loader.appliances
                ]
            )
            status, count = np.unique(ser.sum(axis=0), return_counts=True)
            dic = {"threshold": config["threshold"]["method"], "subset": subset}
            dic.update(dict(zip(status.astype(str), 100 * count / count.sum())))
            dic.update(
                dict(zip(loader.appliances, 100 * ser.sum(axis=1) / count.sum()))
            )
            list_counts.append(pd.DataFrame(dic, index=[0]))
            del loader

        # Update the csv every iteration
        df = pd.concat(list_counts)
        df.to_csv(path_out, index=False)

    # Remove threshold
    os.remove(path_threshold)


if __name__ == "__main__":
    typer.run(main)
