import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from keras.callbacks import EarlyStopping
from torch.utils.data import TensorDataset

from nilm_thresholding.data.thresholding import get_status, get_status_by_duration
from nilm_thresholding.model.export import store_model_json
from nilm_thresholding.utils.scores import (
    classification_scores_dict,
    regression_scores_dict,
)


class KerasModel:
    def __init__(self):
        self.model = None

    def train_with_validation(
        self,
        x_train,
        y_train,
        bin_train,
        x_val,
        y_val,
        bin_val,
        epochs=1000,
        batch_size=64,
        shuffle=False,
        patience=300,
    ):
        """
        Train the model, implementing early stop. The train stops when the
        validation loss ceases to decrease.

        Parameters
        ----------
        x_train : numpy.array
        y_train : numpy.array or list of numpy.array
        bin_train : numpy.array
        x_val : numpy.array
        y_val : numpy.array or list of numpy.array
        bin_val : numpy.array
        epochs : int, default=4000
            Number of epochs to train the model. An epoch is an iteration over
            the entire x and y data provided.
        batch_size : int, default=64
            Number of samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data before each epoch.
        patience : int, default=200
             Number of epochs with no improvement after which training will be
             stopped.
        """
        # patient early stopping
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=patience)

        # Fit model
        self.model.fit(
            x_train,
            [y_train, bin_train],
            validation_data=(x_val, [y_val, bin_val]),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            callbacks=[es],
        )

    def store_json(self, path):
        store_model_json(self.model, path)


class TorchModel:
    model: nn.Module = None
    optimizer: optim.Adam = None
    pow_criterion = nn.MSELoss()
    act_criterion = nn.BCEWithLogitsLoss()

    def __init__(
        self,
        config: dict,
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.input_len = config.get("input_len", 510)
        self.border = config.get("border", 15)
        self.output_len = config.get("output_len", self.input_len - 2 * self.border)
        self._limit = self.border + 1
        self.pow_w = config.get("regression_w", 1)
        self.act_w = config.get("classification_w", 1)
        self.batch_size = config.get("batch_size", 32)
        self.shuffle = config.get("shuffle", True)
        self.pow_loss_avg = config.get("reg_loss_avg", 0.68)
        self.act_loss_avg = config.get("class_loss_avg", 0.0045)
        self.epochs = config.get("epochs", 300)
        self.patience = config.get("patience", 300)
        self.name = config.get("name", "Model")
        self.appliances = config.get("appliances", list())
        self.init_features = config.get("init_features", 32)
        self.dropout = config.get("dropout", 0.1)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.power_scale = config.get("power_scale", 2000)
        self.threshold = config.get("threshold", {})

    def _train_epoch(
        self, train_loader: torch.utils.data.DataLoader, train_losses: list
    ):
        self.model.train()  # prep model for training

        # Initialize ON activation frequency
        # on = np.zeros(3)
        # total = 0

        for batch, (data, target_power, target_status) in enumerate(train_loader, 1):
            data = data.unsqueeze(1).to(device=self.device, dtype=torch.float)
            target_power = target_power.to(device=self.device, dtype=torch.float)
            target_status = target_status.to(device=self.device, dtype=torch.float)

            # clear the gradients of all optimized variables
            self.optimizer.zero_grad()
            # forward pass: compute predicted results by passing inputs
            # to the model
            output_power, output_status = self.model(data)
            output_power = output_power.permute(0, 2, 1)
            output_status = output_status.permute(0, 2, 1)
            # calculate the loss
            pow_loss = self.pow_criterion(output_power, target_power)
            act_loss = self.act_criterion(output_status, target_status)
            loss = (
                self.pow_w * pow_loss / self.pow_loss_avg
                + self.act_w * act_loss / self.act_loss_avg
            )
            # backward pass: compute gradient of the loss with respect
            # to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            self.optimizer.step()
            # record training loss
            train_losses.append(loss.item())
            # Compute ON activation frequency
            # on += target_status.sum(dim=0).sum(dim=0).cpu().numpy()
            # total += target_status.size()[0] * target_status.size()[1]

    def _validation_epoch(
        self, valid_loader: torch.utils.data.DataLoader, valid_losses: list
    ):
        self.model.eval()  # prep model for evaluation
        for data, target_power, target_status in valid_loader:
            data = data.unsqueeze(1).to(device=self.device, dtype=torch.float)
            target_power = target_power.to(device=self.device, dtype=torch.float)
            target_status = target_status.to(device=self.device, dtype=torch.float)

            # forward pass: compute predicted results by passing inputs
            # to the model
            output_power, output_status = self.model(data)
            output_power = output_power.permute(0, 2, 1)
            output_status = output_status.permute(0, 2, 1)
            # calculate the loss
            pow_loss = self.pow_criterion(output_power, target_power)
            act_loss = self.act_criterion(output_status, target_status)
            loss = (
                self.pow_w * pow_loss / self.pow_loss_avg
                + self.act_w * act_loss / self.act_loss_avg
            )
            # record validation loss
            valid_losses.append(loss.item())

    def train_with_dataloader(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
    ):

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        min_loss = np.inf
        loss_up = 0

        for epoch in range(1, self.epochs + 1):

            ###################
            # train the model #
            ###################
            self._train_epoch(train_loader, train_losses)
            # Display ON activation frequency
            # print('Train ON frequency', on / total)

            ######################
            # validate the model #
            ######################
            self._validation_epoch(valid_loader, valid_losses)

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(self.epochs))

            print(
                f"[{epoch:>{epoch_len}}/{self.epochs:>{epoch_len}}] "
                + f"train_loss: {train_loss:.5f} "
                + f"valid_loss: {valid_loss:.5f} "
            )

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # Check if validation loss has decreased
            # If so, store the model as the best model
            if valid_loss < min_loss:
                print(
                    f"Validation loss decreased ({min_loss:.6f} -->"
                    f" {valid_loss:.6f}).  Saving model ..."
                )
                min_loss = valid_loss
                self.save("model.pth")
            else:
                loss_up += 1

            if loss_up >= self.patience:
                break

        # Take best model
        # load the last checkpoint with the best model
        self.load("model.pth")
        os.remove("model.pth")

    def predict(self, loader: torch.utils.data.DataLoader):
        x_true = []
        s_true = []
        p_true = []
        s_hat = []
        p_hat = []

        self.model.eval()

        with torch.no_grad():
            for x, power, status in loader:
                x = x.unsqueeze(1).to(device=self.device, dtype=torch.float)

                pw, sh = self.model(x)
                sh = torch.sigmoid(sh)

                sh = sh.permute(0, 2, 1)
                sh = sh.detach().cpu().numpy()
                s_hat.append(sh.reshape(-1, sh.shape[-1]))

                pw = pw.permute(0, 2, 1)
                pw = pw.detach().cpu().numpy()
                p_hat.append(pw.reshape(-1, pw.shape[-1]))

                x_true.append(
                    x[:, :, self._limit : -self._limit].detach().cpu().numpy().flatten()
                )
                s_true.append(status.detach().cpu().numpy().reshape(-1, sh.shape[-1]))
                p_true.append(power.detach().cpu().numpy().reshape(-1, sh.shape[-1]))

        x_true = np.hstack(x_true)
        s_true = np.concatenate(s_true, axis=0)
        p_true = np.concatenate(p_true, axis=0)
        s_hat = np.concatenate(s_hat, axis=0)
        p_hat = np.concatenate(p_hat, axis=0)

        return x_true, p_true, s_true, p_hat, s_hat

    def save(self, path_model: str):
        """Store the weights of the model"""
        torch.save(self.model.state_dict(), path_model)

    def load(self, path_model: str):
        """Load the weights of the model"""
        self.model.load_state_dict(torch.load(path_model))

    def process_outputs(self, p_true, p_hat, s_hat):
        # Denormalize power values
        p_true = np.multiply(p_true, self.power_scale)
        p_hat = np.multiply(p_hat, self.power_scale)
        p_hat[p_hat < 0.0] = 0.0

        # Get status
        if (self.threshold["min_on"] is None) or (self.threshold["min_off"] is None):
            s_hat[s_hat >= 0.5] = 1
            s_hat[s_hat < 0.5] = 0
        else:
            thresh = [0.5] * len(self.threshold["min_on"])
            s_hat = get_status_by_duration(
                s_hat,
                thresh,
                self.threshold["min_off"],
                self.threshold["min_on"],
            )

        # Get power values from status
        sp_hat = np.multiply(np.ones(s_hat.shape), self.means[:, 0])
        sp_on = np.multiply(np.ones(s_hat.shape), self.means[:, 1])
        sp_hat[s_hat == 1] = sp_on[s_hat == 1]

        # Get status from power values
        ps_hat = get_status(p_hat, self.thresholds)

        return p_true, p_hat, s_hat, sp_hat, ps_hat

    def get_scores(self, loader: torch.utils.data.DataLoader):
        """
        Returns its activation and power scores.
        """

        # Test
        x_true, p_true, s_true, p_hat, s_hat = self.predict(loader)

        p_true, p_hat, s_hat, sp_hat, ps_hat = self.process_outputs(
            p_true, p_hat, s_hat
        )

        # classification scores

        class_scores = classification_scores_dict(s_hat, s_true, self.appliances)
        reg_scores = regression_scores_dict(sp_hat, p_true, self.appliances)
        act_scores = [class_scores, reg_scores]

        print("classification scores")
        print(class_scores)
        print(reg_scores)

        # regression scores

        class_scores = classification_scores_dict(ps_hat, s_true, self.appliances)
        reg_scores = regression_scores_dict(p_hat, p_true, self.appliances)
        pow_scores = [class_scores, reg_scores]

        print("regression scores")
        print(class_scores)
        print(reg_scores)

        return act_scores, pow_scores
