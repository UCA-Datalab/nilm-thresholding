from itertools import combinations
from typing import List, Tuple

import pandas as pd
import typer

from nilmth.data.dataloader import DataLoader
from nilmth.generate_config_files import LIST_MODELS
from nilmth.utils.config import load_config
from nilmth.utils.model import initialize_model


def main(
    path_config: str = "./nilmth/config.toml",
    path_data: str = "./data-prep",
    path_out: str = "computational_time.csv",
):
    """
    Compute the training time of models with given config,
    for each possible combination of appliances and for each model

    Parameters
    ----------
    path_config : str, optional
        Path to the config file, by default "./nilmth/config.toml"
    path_data : str, optional
        Path to the processed data, by default "./data-prep"
    path_out : str, optional
        Path where the csv is stored, by default "computational_time.csv"
    """
    config = load_config(path_config, "model")
    list_appliances = config["appliances"]
    list_combinations: List[Tuple[str]] = []
    for n in range(1, len(list_appliances) + 1):
        list_combinations += list(combinations(list_appliances, n))
    list_df: List[pd.DataFrame] = []
    for app in list_combinations:
        for model in LIST_MODELS:
            config.update({"name": model, "appliances": list(app)})
            # Load dataloader
            dataloader_train = DataLoader(
                path_data, subset="train", shuffle=True, **config
            )
            dataloader_validation = DataLoader(
                path_data, subset="validation", shuffle=True, **config
            )
            model = initialize_model(config)
            # Train
            time_elapsed = model.train(dataloader_train, dataloader_validation)
            df = pd.DataFrame({"appliances": app, "model": model, "time": time_elapsed})
            list_df.append(df)
            df = pd.concat(list_df)
            df.to_csv(path_out)


if __name__ == "__main__":
    typer.run(main)
