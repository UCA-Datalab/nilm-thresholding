import toml
import os
import typer

from better_nilm.utils.conf import load_conf


LIST_METHODS = ["mp", "vs", "at"]
LIST_MODELS = ["ConvModel", "GRUModel"]
LIST_CLASS_W = [
    0,
    0.01,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.99,
    1,
]


def main(path_config="better_nilm/config.toml", path_output="configs"):
    config = load_conf(path_config)

    if not os.path.exists(path_output):
        os.mkdir(path_output)

    for model_name in LIST_MODELS:
        config["train"].update({"name": model_name})
        for method in LIST_METHODS:
            config["data"]["threshold"].update({"method": method})
            for class_w in LIST_CLASS_W:
                config["train"]["model"].update({"classification_w": class_w})
                path_dump = os.path.join(
                    path_output, f"{model_name}_{method}_classw_{class_w}.toml"
                )
                with open(path_dump, "w") as toml_file:
                    toml.dump(config, toml_file)


if __name__ == "__main__":
    typer.run(main)
