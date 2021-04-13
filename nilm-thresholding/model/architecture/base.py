import os

import numpy as np
import torch
from keras.callbacks import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader

from nilm-thresholding.model.export import store_model_json


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

    def predict(self, x_test):
        return self.model.predict(x_test)

    def store_json(self, path):
        store_model_json(self.model, path)


class TorchModel:
    def __init__(self, batch_size=32):
        self.model = None
        self.batch_size = batch_size
        self.shuffle = True
        self.pow_w = 1
        self.act_w = 1
        self.border = 0
        self.pow_loss_avg = 0.0045
        self.act_loss_avg = 0.68

    def _get_dataloader(self, x, y, y_bin):
        tensor_x = torch.Tensor(x)
        tensor_y = torch.Tensor(y)
        tensor_bin = torch.Tensor(y_bin)
        dataset = TensorDataset(tensor_x, tensor_y, tensor_bin)
        data_loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
        return data_loader

    def train_with_dataloader(
        self, train_loader, valid_loader, epochs=1000, patience=300
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

        for epoch in range(1, epochs + 1):

            ###################
            # train the model #
            ###################
            self.model.train()  # prep model for training

            # Initialize ON activation frequency
            # on = np.zeros(3)
            # total = 0

            for batch, (data, target_power, target_status) in enumerate(
                train_loader, 1
            ):
                data = data.unsqueeze(1).cuda()
                target_power = target_power.cuda()
                target_status = target_status.cuda()

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

            # Display ON activation frequency
            # print('Train ON frequency', on / total)

            ######################
            # validate the model #
            ######################
            self.model.eval()  # prep model for evaluation
            for data, target_power, target_status in valid_loader:
                data = data.unsqueeze(1).cuda()
                target_power = target_power.cuda()
                target_status = target_status.cuda()

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

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(epochs))

            print_msg = (
                f"[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] "
                + f"train_loss: {train_loss:.5f} "
                + f"valid_loss: {valid_loss:.5f} "
            )

            print(print_msg)

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

            if loss_up >= patience:
                break

        # Take best model
        # load the last checkpoint with the best model
        self.load("model.pth")
        os.remove("model.pth")

    def train_with_data(
        self,
        x_train,
        y_train,
        bin_train,
        x_val,
        y_val,
        bin_val,
        epochs=1000,
        batch_size=32,
        shuffle=False,
        patience=300,
    ):

        self.batch_size = batch_size
        self.shuffle = shuffle

        train_loader = self._get_dataloader(x_train, y_train, bin_train)
        valid_loader = self._get_dataloader(x_val, y_val, bin_val)

        self.train_with_dataloader(
            train_loader, valid_loader, epochs=epochs, patience=patience
        )

    def predict(self, x_test):
        self.model.eval()
        tensor_x = torch.Tensor(x_test)
        tensor_x = tensor_x.permute(0, 2, 1).cuda()
        output_power, output_status = self.model(tensor_x)
        output_power = output_power.permute(0, 2, 1)
        output_status = output_status.permute(0, 2, 1)
        return output_power, output_status

    def predict_loader(self, loader):
        x_true = []
        s_true = []
        p_true = []
        s_hat = []
        p_hat = []

        self.model.eval()

        with torch.no_grad():
            for x, power, status in loader:
                x = x.unsqueeze(1).cuda()

                pw, sh = self.model(x)
                sh = torch.sigmoid(sh)

                sh = sh.permute(0, 2, 1)
                sh = sh.detach().cpu().numpy()
                s_hat.append(sh.reshape(-1, sh.shape[-1]))

                pw = pw.permute(0, 2, 1)
                pw = pw.detach().cpu().numpy()
                p_hat.append(pw.reshape(-1, pw.shape[-1]))

                x_true.append(
                    x[:, :, self.border : -self.border].detach().cpu().numpy().flatten()
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
