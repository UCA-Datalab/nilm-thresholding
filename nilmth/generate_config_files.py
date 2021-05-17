import toml
import os
import typer

from nilmth.utils.config import load_config


LIST_METHODS = ["mp", "vs", "at"]
LIST_MODELS = ["ConvModel", "GRUModel"]
LIST_CLASS_W = [
    0,
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
    1,
]


def main(path_config="nilmth/config.toml", path_output="configs"):
    config = load_config(path_config)

    if not os.path.exists(path_output):
        os.mkdir(path_output)

    for model_name in LIST_MODELS:
        config["model"].update({"name": model_name})
        for method in LIST_METHODS:
            config["model"]["threshold"].update({"method": method})
            for class_w in LIST_CLASS_W:
                config["model"].update({"classification_w": class_w})
                config["model"].update({"regression_w": round(1 - class_w, 2)})
                path_dump = os.path.join(
                    path_output, f"{model_name}_{method}_classw_{class_w}.toml"
                )
                with open(path_dump, "w") as toml_file:
                    toml.dump(config, toml_file)


if __name__ == "__main__":
    typer.run(main)
