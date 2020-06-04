from better_nilm.model.train import train_with_validation
from better_nilm.model.export import store_model_json


class BaseModel:
    def __init__(self):
        self.model = None

    def train_with_validation(self, x_train, y_train, bin_train,
                              x_val, y_val, bin_val,
                              epochs=1000, batch_size=64,
                              shuffle=False, patience=300):
        self.model = train_with_validation(self.model,
                                           x_train, [y_train, bin_train],
                                           x_val, [y_val, bin_val],
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           patience=patience)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def store_json(self, path):
        store_model_json(self.model, path)
