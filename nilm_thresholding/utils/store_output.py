import collections
import os


def list_scores(appliances, act_scores, pow_scores, num_models):
    """
    List scores in dictionary format.
    """
    scores = {"classification": {}, "regression": {}}

    for app in appliances:
        counter = collections.Counter()
        for sc in act_scores:
            counter.update(sc[app])
        scores["classification"][app] = {
            k: round(v, 6) / num_models for k, v in dict(counter).items()
        }

        counter = collections.Counter()
        for sc in pow_scores:
            counter.update(sc[app])
        scores["regression"][app] = {
            k: round(v, 6) / num_models for k, v in dict(counter).items()
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
    path_output: str,
    config_model: dict,
    dict_scores: dict,
    time_elapsed: float = 0,
    filename: str = "scores.txt",
):
    """

    Parameters
    ----------
    path_output : str
    config_model : dict
    dict_scores : dict
        Output of utils.scores.score_dict_predictions
    time_elapsed : float
    filename : str

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

    path_scores = os.path.join(path_output, filename)

    with open(path_scores, "w") as text_file:
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
