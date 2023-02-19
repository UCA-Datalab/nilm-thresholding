import os
from itertools import combinations
from typing import List, Tuple

import pandas as pd
import typer

from nilmth.data.dataloader import DataLoader
from nilmth.data.temporal import generate_temporal_data, remove_directory
from nilmth.generate_config_files import LIST_MODELS
from nilmth.utils.config import load_config
from nilmth.utils.model import initialize_model


def main(
    epochs: int = 20,
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
    # List all possible combination of appliances
    list_appliances = config["appliances"]
    list_combinations: List[Tuple[str]] = []
    for n in range(1, len(list_appliances) + 1):
        list_combinations += list(combinations(list_appliances, n))
    # Initialize the list of dataframes
    list_df: List[pd.DataFrame] = []
    # Temporal threshold file
    path_threshold = "threshold_comp.toml"
    # Loop over the appliance combinations and models
    for name in LIST_MODELS:
        for app in list_combinations:
            # Update the configuration with these
            config.update({"name": name, "appliances": list(app), "epochs": epochs})
            # Load dataloader
            dataloader_train = DataLoader(
                path_data,
                subset="train",
                shuffle=True,
                path_threshold=path_threshold,
                **config
            )
            dataloader_validation = DataLoader(
                path_data,
                subset="validation",
                shuffle=True,
                path_threshold=path_threshold,
                **config
            )

            generate_temporal_data(dataloader_train, path="temp_train")
            generate_temporal_data(dataloader_validation, path="temp_valid")
            # Initialize and train the model
            model = initialize_model(config)
            time_elapsed = model.train(dataloader_train, dataloader_validation)

            remove_directory("temp_train")
            remove_directory("temp_valid")
            # Store results in the list of dataframes
            df = pd.DataFrame(
                {
                    "appliances": "+".join(app),
                    "amount": len(app),
                    "model": name,
                    "time": time_elapsed,
                },
                index=[0],
            )
            list_df.append(df)
            # Concatenate the list and output the list after every iteration
            df = pd.concat(list_df)
            df.to_csv(path_out, index=False)

            # Remove threshold
            os.remove(path_threshold)


if __name__ == "__main__":
    typer.run(main)
