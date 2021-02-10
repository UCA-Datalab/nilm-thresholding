import toml
import os
import typer

from better_nilm.utils.conf import load_conf


def main(path_config="better_nilm/config.toml", path_output="configs"):
    config = load_conf(path_config)
    config_data = config["data"]
    config_threshold = config_data["threshold"]
    config_train = config["train"]
    config_model = config_train["model"]

    if not os.path.exists(path_output):
        os.mkdir(path_output)

    for method in ["mp", "vs", "at"]:
        config_threshold.update({"method": method})
        config_data.update({"threshold": config_threshold})
        config.update({"data": config_data})
        for class_w in [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]:
            config_model.update({"classification_w": class_w})
            config_train.update({"model": config_model})
            config.update({"train": config_train})
            path_dump = os.path.join(path_output, f"{method}_classw_{class_w}.toml")
            with open(path_dump, "w") as toml_file:
                toml.dump(config, toml_file)


if __name__ == "__main__":
    typer.run(main)
