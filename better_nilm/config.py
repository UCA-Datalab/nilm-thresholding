DATA_TRAIN = {
    "redd": [1]
}

APPLIANCES = {
    "dish_washer": 1,
    "microwave": 0
}

MODEL = {
    "sample_period": 6,
    "window_size": 600,
    "hidden_size": 100,
    "lstm_layers": 1,
    "output": "regression"
}

PARAMS_TRAIN = {
    "step": 100,
    "train_size": .8,
    "val_size": .1,
    "epochs": 100,
    "val_epochs": 10,
    "early_stop": 100,
    "batch_size": 128,
    "learning_rate": .005,
    "dropout": .25
}

DATA_TEST = {
    "redd": [1]
}

PARAMS_TEST = {
    "skip_size": .9,
    "step": 100
}
