import typer

from nilmth.utils.config import load_config
from nilmth.utils.model import test_many_models


def main(
    path_data: str = "data-prep",
    path_output: str = "outputs",
    path_config: str = "nilmth/config.toml",
):
    """
    Trains several CONV models under the same conditions
    Stores scores and plots on results folder

    Parameters
    ----------
    path_data : str, optional
        Path to UK-DALE data
    path_output : str, optional
        Path to the results folder
    path_config : str, optional
        Path to the config toml file
    """

    print(f"\nLoading config file from {path_config}")
    # Load config file
    config = load_config(path_config, "model")
    print("Done\n")

    # Run main results
    print(f"{config['name']}\n")

    test_many_models(path_data, path_output, config)


if __name__ == "__main__":
    typer.run(main)
