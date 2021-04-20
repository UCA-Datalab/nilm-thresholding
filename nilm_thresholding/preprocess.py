import typer

from nilm_thresholding.data.ukdale import UkdalePreprocess
from nilm_thresholding.utils.config import load_config


def main(
    path_data: str = "data/ukdale",
    path_output: str = "data-prep",
    path_config: str = "nilm_thresholding/config.toml",
):
    """
    Trains several CONV models under the same conditions
    Stores scores and plots on results folder

    Parameters
    ----------
    path_data : str, optional
        Path to data
    path_output : str, optional
        Path to the results folder
    path_config : str, optional
        Path to the config toml file
    """

    print(f"\nLoading config file from {path_config}")
    # Load config file
    config = load_config(path_config, "model")
    print("Done\n")

    path_h5 = path_data + ".h5"

    dataloader = UkdalePreprocess(path_h5, path_data, config)
    dataloader.store_preprocessed_data(path_output)


if __name__ == "__main__":
    typer.run(main)
