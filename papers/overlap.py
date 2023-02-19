from pathlib import Path

import numpy as np
import pandas as pd

from nilmth.data.dataloader import DataLoader
from nilmth.utils.config import load_config

path_configs = Path("./configs")
# Initialize the list that will contain the overlaps for each configuration
list_counts = []

# Loop over all possible configurations
for path_config in path_configs.iterdir():
    name = path_config.stem  # Name of the configuration
    config = load_config(path_config, "model")
    # Loop over both subsets
    for subset in ["train", "test"]:
        loader = DataLoader(
            "./data-prep/", subset=subset, path_threshold=path_config, **config
        )

        ser = np.stack(
            [loader.get_appliance_series(app, "status") for app in loader.appliances]
        )
        status, count = np.unique(ser.sum(axis=0), return_counts=True)
        list_counts.append(
            pd.Series(100 * count / count.sum(), index=status, name=f"{name}_{subset}")
        )

    # Update the csv every iteration
    df = pd.DataFrame(list_counts)
    df.to_csv("overlap.csv")
