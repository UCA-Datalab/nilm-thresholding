import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from nilm_thresholding.data.threshold import Threshold
from nilm_thresholding.utils.logging import logger
from nilm_thresholding.utils.scores import (
    classification_scores_dict,
    regression_scores_dict,
)


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
        self._limit = self.border + 1
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
        self.appliances = [] if appliances is None else appliances
        self.init_features = init_features
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.power_scale = power_scale

        # Set the parameters according to given threshold method
        param_thresh = {} if threshold is None else threshold
        self.threshold = Threshold(appliances=self.appliances, **param_thresh)

        logger.debug(f"Received extra kwargs, not used:\n   {', '.join(kwargs.keys())}")

    def _train_epoch(self, train_loader: torch.utils.data.DataLoader) -> np.array:
        # Initialize list of train losses and set model to train mode
        train_losses = []
        self.model.train()  # prep model for training

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
        return np.average(train_losses)

    def _validation_epoch(self, valid_loader: torch.utils.data.DataLoader) -> np.array:
        valid_losses = []
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
        return np.average(valid_losses)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
    ):

        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        min_loss = np.inf
        loss_up = 0

        for epoch in range(1, self.epochs + 1):
            # Train and validate the model
            train_loss = self._train_epoch(train_loader)
            valid_loss = self._validation_epoch(valid_loader)

            # print training/validation statistics
            # calculate average loss over an epoch
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

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
                sh = torch.sigmoid(sh).permute(0, 2, 1).detach().cpu().numpy()
                s_hat.append(sh.reshape(-1, sh.shape[-1]))

                pw = pw.permute(0, 2, 1).detach().cpu().numpy()
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
        s_hat = self.threshold.get_status(s_hat)

        # Get power values from status
        sp_hat = np.multiply(np.ones(s_hat.shape), self.threshold.centroids[:, 0])
        sp_on = np.multiply(np.ones(s_hat.shape), self.threshold.centroids[:, 1])
        sp_hat[s_hat == 1] = sp_on[s_hat == 1]

        # Get status from power values
        ps_hat = self.threshold.get_status(p_hat)

        return p_true, p_hat, s_hat, sp_hat, ps_hat

    def score(self, loader: torch.utils.data.DataLoader):
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

        # regression scores
        class_scores = classification_scores_dict(ps_hat, s_true, self.appliances)
        reg_scores = regression_scores_dict(p_hat, p_true, self.appliances)
        pow_scores = [class_scores, reg_scores]

        return act_scores, pow_scores
