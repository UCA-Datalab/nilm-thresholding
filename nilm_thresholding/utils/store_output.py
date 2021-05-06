import collections
import os


def average_list_dict_scores(list_dict_scores: list) -> dict:
    """
    Averages a list of dictionaries with the same format to a single dictionary
    """
    num_models = len(list_dict_scores)
    appliances = list(list_dict_scores[0]["classification"].keys())
    scores = {"classification": {}, "regression": {}}

    for app in appliances:
        counter_class = collections.Counter()
        counter_reg = collections.Counter()
        for sc in list_dict_scores:
            counter_class.update(sc["classification"][app])
            counter_reg.update(sc["regression"][app])
        scores["classification"][app] = {
            k: round(v, 6) / num_models for k, v in dict(counter_class).items()
        }
        scores["regression"][app] = {
            k: round(v, 6) / num_models for k, v in dict(counter_reg).items()
        }

    return scores


def generate_path_output(path_output, model_name):
    """
    Generates results folder.
    """
    if not os.path.isdir(path_output):
        os.mkdir(path_output)

    path_output = os.path.join(path_output, model_name)
    if not os.path.isdir(path_output):
        os.mkdir(path_output)

    return path_output


def generate_folder_name(path_output: str, config: dict):
    """
    Generates specific folder inside results.
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
