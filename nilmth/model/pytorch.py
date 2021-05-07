import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from nilmth.data.dataloader import DataLoader
from nilmth.data.threshold import Threshold
from nilmth.utils.logging import logger


class TorchModel:
    model: nn.Module = None
    optimizer: optim.Adam = None
    pow_criterion = nn.MSELoss()
    act_criterion = nn.BCEWithLogitsLoss()

    def __init__(
        self,
        border: int = 15,
        input_len: int = 510,
        regression_w: float = 1,
        classification_w: float = 1,
        batch_size: int = 32,
        shuffle: bool = True,
        reg_loss_avg: float = 0.68,
        class_loss_avg: float = 0.0045,
        name: str = "Model",
        epochs: int = 300,
        patience: int = 300,
        appliances: list = None,
        init_features: int = 32,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        power_scale: int = 2000,
        threshold: dict = None,
        **kwargs,
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.debug(f"Using device: {self.device}")

        # Parameters expected to be found in the configuration dictionary
        self.border = border
        self._limit = self.border
        self.input_len = input_len
        self.output_len = self.input_len - 2 * self.border
        self.pow_w = regression_w
        self.act_w = classification_w
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pow_loss_avg = reg_loss_avg
        self.act_loss_avg = class_loss_avg
        self.epochs = epochs
        self.patience = patience
        self.name = name
        self.appliances = [] if appliances is None else sorted(appliances)
        self.status = [app + "_status" for app in self.appliances]
        self.num_apps = len(self.appliances)
        self.init_features = init_features
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.power_scale = power_scale

        # Set the parameters according to given threshold method
        param_thresh = {} if threshold is None else threshold
        self.threshold = Threshold(appliances=self.appliances, **param_thresh)

        logger.debug(f"Received extra kwargs, not used:\n   {', '.join(kwargs.keys())}")

    def _normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes X data by subtracting the mean of each series and dividing by a
        constant power value

        Parameters
        ----------
        x : torch.Tensor
            shape [batch, 1, input len]

        Returns
        -------
        torch.Tensor
            shape [batch, 1, input len]

        """
        # x data are subtracted their mean and normalized
        return (x - x.mean(axis=2, keepdim=True)) / self.power_scale

    def _normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return y / self.power_scale

    def _denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        y[y < 0] = 0
        return y * self.power_scale

    def _process_loader_data(self, data: tuple) -> tuple:
        """Reads the data provided by the data loader and processes it

        Parameters
        ----------
        data : tuple (torch.Tensor)
            x : shape [batch, input len]
            y_pow : shape [batch, output len, num appliances]
            y_sta : shape [batch, output len, num appliances]

        Returns
        -------
        tuple (torch.Tensor)
            x : shape [batch, 1, input len]
            y_pow : shape [batch, num appliances, output len]
            y_sta : shape [batch, num appliances, output len]

        """
        x, y_pow, y_sta = data
        x = x.unsqueeze(1).to(device=self.device, dtype=torch.float)
        x = self._normalize_x(x)
        # Permute y arrays
        y_pow = self._normalize_y(
            y_pow.to(device=self.device, dtype=torch.float).permute(0, 2, 1)
        )
        y_sta = y_sta.to(device=self.device, dtype=torch.float).permute(0, 2, 1)
        return x, y_pow, y_sta

    def _loader_data_to_numpy(self, data: tuple) -> tuple:
        """Reads the data provided by the data loader, moves it to CPU, transforms to
        numpy and reshapes it to ignore batch

        Parameters
        ----------
        data : tuple (torch.Tensor)
            x : shape [batch, input len]
            y_pow : shape [batch, output len, num appliances]
            y_sta : shape [batch, output len, num appliances]

        Returns
        -------
        tuple (numpy.array)
            x : shape [batch * output len]
            y_pow : shape [batch * output len, num appliances]
            y_sta : shape [batch * output len, num appliances]

        """
        x_raw, y_pow, y_sta = data
        # (batch_size, out_len) -> (batch_size * out_len)
        aggregated = (
            x_raw[:, self._limit : -self._limit].detach().cpu().numpy().flatten()
        )
        # (batch_size, out_len, num_apps) -> (batch_size * out_len, num_apps)
        status_true = y_sta.detach().cpu().numpy().reshape(-1, self.num_apps)
        power_true = y_pow.detach().cpu().numpy().reshape(-1, self.num_apps)
        return aggregated, power_true, status_true

    def _compute_loss(
        self,
        output_power: torch.Tensor,
        y_pow: torch.Tensor,
        output_status: torch.Tensor,
        y_sta: torch.Tensor,
    ):
        """All input tensors must have shape [batch, num appliances, output len]"""
        pow_loss = self.pow_criterion(output_power, y_pow)
        act_loss = self.act_criterion(output_status, y_sta)

        loss = (
            self.pow_w * pow_loss / self.pow_loss_avg
            + self.act_w * act_loss / self.act_loss_avg
        )
        return loss

    def _train_epoch(self, train_loader: DataLoader) -> np.array:
        # Initialize list of train losses and set model to train mode
        train_losses = [0] * len(train_loader)
        self.model.train()  # prep model for training

        for batch, data in enumerate(train_loader):
            x, y_pow, y_sta = self._process_loader_data(data)
            # clear the gradients of all optimized variables
            self.optimizer.zero_grad()
            # forward pass: compute predicted results by passing inputs
            # to the model
            output_power, output_status = self.model(x)
            # calculate the loss
            loss = self._compute_loss(output_power, y_pow, output_status, y_sta)
            # backward pass: compute gradient of the loss with respect
            # to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            self.optimizer.step()
            # record training loss
            train_losses[batch] = loss.item()
        return np.average(train_losses)

    def _validation_epoch(self, valid_loader: DataLoader) -> np.array:
        valid_losses = [0] * len(valid_loader)
        self.model.eval()  # prep model for evaluation
        for batch, data in enumerate(valid_loader):
            x, y_pow, y_sta = self._process_loader_data(data)
            # forward pass: compute predicted results by passing inputs
            # to the model
            output_power, output_status = self.model(x)
            # calculate the loss
            loss = self._compute_loss(output_power, y_pow, output_status, y_sta)
            # record validation loss
            valid_losses[batch] = loss.item()
        return np.average(valid_losses)

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ):
        """Trains the model

        Parameters
        ----------
        train_loader : DataLoader
        valid_loader : DataLoader

        Returns
        ------
        float
            elapsed time, in seconds

        """
        # Time it
        time_start = time.time()

        # to track the average training loss per epoch as the model trains
        avg_train_losses = [0] * self.epochs
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = [0] * self.epochs

        min_loss = np.inf
        loss_up = 0

        for epoch in range(1, self.epochs + 1):
            # Train and validate the model
            train_loss = self._train_epoch(train_loader)
            valid_loss = self._validation_epoch(valid_loader)

            # print training/validation statistics
            # calculate average loss over an epoch
            avg_train_losses[epoch - 1] = train_loss
            avg_valid_losses[epoch - 1] = valid_loss

            epoch_len = len(str(self.epochs))

            logger.info(
                f"[{epoch:>{epoch_len}}/{self.epochs:>{epoch_len}}] "
                f"train_loss: {train_loss:.5f} "
                f"valid_loss: {valid_loss:.5f}"
            )

            # Check if validation loss has decreased
            # If so, store the model as the best model
            if valid_loss < min_loss:
                logger.info(
                    f"Validation loss decreased ({min_loss:.6f} -->"
                    f" {valid_loss:.6f}).  Saving model ..."
                )
                min_loss = valid_loss
                self.save("model.pth")
                # Reset patience count
                loss_up = 0
            else:
                loss_up += 1
                if loss_up >= self.patience:
                    break

        # Take best model
        # load the last checkpoint with the best model
        self.load("model.pth")
        os.remove("model.pth")

        # Return time (seconds)
        time_elapsed = round(time.time() - time_start, 2)
        return time_elapsed

    def predict(self, x: torch.Tensor) -> tuple:
        """

        Parameters
        ----------
        x : torch.Tensor
            shape [batch size, input len]

        Returns
        -------
        tuple (numpy.array)
            power_predict : shape [batch size * output len, num appliances]
            status_predict :shape [batch size * output len, num appliances]

        """
        self.model.eval()
        pred_power, pred_status = self.model(x)

        status_predict = (
            # torch.sigmoid(pred_status)
            pred_status.permute(0, 2, 1)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, self.num_apps)
        )
        # Status to integer
        status_predict = status_predict.astype(int)
        status_predict[status_predict < 0] = 0
        n = self.threshold.num_status - 1
        status_predict[status_predict > n] = n

        power_predict = (
            self._denormalize_y(pred_power)
            .permute(0, 2, 1)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, self.num_apps)
        )
        power_predict[power_predict < 0] = 0
        return power_predict, status_predict

    def predictions_to_dictionary(self, loader: DataLoader) -> dict:
        """Builds a dictionary with the following structure:
        {
            "aggregated": series of aggregated power load,
            "appliance": {
                "power": series of appliance power load,
                "status": series of appliance status,
                "power_pred": series of predicted appliance power load,
                "status_pred": series of predicted appliance status
            }
        }
        """
        # Initialize arrays
        aggregated = [0.0] * len(loader)
        status_true = [0] * len(loader)
        power_true = [0.0] * len(loader)
        status_predict = [0] * len(loader)
        power_predict = [0.0] * len(loader)

        with torch.no_grad():
            for batch, data in enumerate(loader):
                # Move torch tensors to CPU, change to numpy and reshape
                (
                    aggregated[batch],
                    power_true[batch],
                    status_true[batch],
                ) = self._loader_data_to_numpy(data)

                x, _, _ = self._process_loader_data(data)

                power_predict[batch], status_predict[batch] = self.predict(x)

        dict_series = {"aggregated": np.hstack(aggregated)}
        # True values
        status_true = np.concatenate(status_true, axis=0)
        power_true = np.concatenate(power_true, axis=0)
        status_predict = np.concatenate(status_predict, axis=0)
        power_predict = np.concatenate(power_predict, axis=0)
        # Reconstructed power, and status from predicted power
        power_reconstructed = loader.dataset.status_to_power(status_predict)
        status_from_power = loader.dataset.power_to_status(power_predict)
        for idx, app in enumerate(self.appliances):
            dict_series.update(
                {
                    app: {
                        "power": power_true[:, idx],
                        "status": status_true[:, idx],
                        "power_pred": power_predict[:, idx],
                        "status_pred": status_predict[:, idx],
                        "power_recon": power_reconstructed[:, idx],
                        "status_from_power": status_from_power[:, idx],
                    }
                }
            )

        return dict_series

    def save(self, path_model: str):
        """Store the weights of the model"""
        torch.save(self.model.state_dict(), path_model)

    def load(self, path_model: str):
        """Load the weights of the model"""
        self.model.load_state_dict(torch.load(path_model))
