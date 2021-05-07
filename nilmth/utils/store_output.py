import os

from nilmth.utils.plot import plot_real_data, plot_real_vs_prediction


def generate_path_output_model(path_output: str, model_name: str) -> str:
    """
    Generates the output folder for a certain model, inside path_output
    """
    if not os.path.isdir(path_output):
        os.mkdir(path_output)

    path_output = os.path.join(path_output, model_name)
    if not os.path.isdir(path_output):
        os.mkdir(path_output)

    return path_output


def generate_path_output_model_params(path_output: str, config: dict) -> str:
    """
    Generates the output folder for a certain model configuration, inside path_output.
    """
    input_len = config["input_len"]
    period = config["period"]
    class_w = config["classification_w"]
    reg_w = config["regression_w"]
    threshold_method = config["threshold"]["method"]

    name = (
        f"seq_{str(input_len)}_{period}_{threshold_method}"
        f"_clas_{str(int(class_w * 100))}"
        f"_reg_{str(int(reg_w * 100))}"
    )
    path_output = os.path.join(path_output, name)
    if not os.path.isdir(path_output):
        os.mkdir(path_output)

    return path_output


def store_scores(
    config_model: dict,
    dict_scores: dict,
    time_elapsed: float = 0,
    path_output: str = "scores.txt",
):
    """

    Parameters
    ----------
    config_model : dict
    dict_scores : dict
        Output of utils.scores.score_dict_predictions
    time_elapsed : float, optional
    path_output : str, optional

    """
    # Load parameters
    class_w = config_model["classification_w"]
    reg_w = config_model["regression_w"]
    train_size = config_model["train_size"]
    valid_size = config_model["valid_size"]
    num_models = config_model["num_models"]
    batch_size = config_model["batch_size"]
    learning_rate = config_model["learning_rate"]
    dropout = config_model["dropout"]
    epochs = config_model["epochs"]
    patience = config_model["patience"]

    with open(path_output, "w") as text_file:
        text_file.write(
            f"Train size: {train_size}\n"
            f"Validation size: {valid_size}\n"
            f"Number of models: {num_models}\n"
            f"Batch size: {batch_size}\n"
            f"Learning rate: {learning_rate}\n"
            f"Dropout: {dropout}\n"
            f"Epochs: {epochs}\n"
            f"Patience: {patience}\n"
            f"Time per model (seconds): {time_elapsed}\n"
            f"=============================================\n"
        )
        for key, dic1 in dict_scores.items():

            # Skip scores if weight is zero
            if (class_w == 0) and (key.startswith("class")):
                continue
            if (reg_w == 0) and (key.startswith("reg")):
                continue

            text_file.write(f"{key}\n------------------------------------------\n")
            for app, dic2 in dic1.items():
                text_file.write(f"{app} \n")
                for name, value in dic2.items():
                    text_file.write(f"{name}: {value}\n")
                text_file.write("----------------------------------------------\n")
            text_file.write("==================================================\n")


def store_plots(dict_pred: dict, path_output: str = "."):
    plot_real_data(dict_pred, savefig=os.path.join(path_output, "real_data.png"))
    # List appliances
    list_app = list(dict_pred.keys())
    list_app.remove("aggregated")
    for app in list_app:
        plot_real_vs_prediction(
            dict_pred, app, savefig=os.path.join(path_output, f"{app}.png")
        )
