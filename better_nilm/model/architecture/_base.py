import numpy as np
import torch
from keras.callbacks import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader

from better_nilm.model.export import store_model_json


class KerasModel:
    def __init__(self):
        self.model = None

    def train_with_validation(self, x_train, y_train, bin_train,
                              x_val, y_val, bin_val,
                              epochs=1000, batch_size=64,
                              shuffle=False, patience=300):
        """
        Train the model, implementing early stop. The train stops when the
        validation loss ceases to decrease.

        Parameters
        ----------
        x_train : numpy.array
        y_train : numpy.array or list of numpy.array
        x_val : numpy.array
        y_val : numpy.array or list of numpy.array
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
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                           patience=patience)

        # Fit model
        self.model.fit(x_train, [y_train, bin_train],
                       validation_data=(x_val, [y_val, bin_val]),
                       epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                       callbacks=[es])

    def predict(self, x_test):
        return self.model.predict(x_test)

    def store_json(self, path):
        store_model_json(self.model, path)


class TorchModel:
    def __init__(self):
        self.model = None
        self.batch_size = 64
        self.shuffle = True

    def _get_dataloader(self, x, y, y_bin):
        tensor_x = torch.Tensor(x)
        tensor_y = torch.Tensor([y, y_bin])
        dataset = TensorDataset(tensor_x, tensor_y)
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle)
        return data_loader

    def train_with_validation(self, x_train, y_train, bin_train,
                              x_val, y_val, bin_val,
                              epochs=1000, batch_size=64,
                              shuffle=False, patience=300):

        self.batch_size = batch_size
        self.shuffle = shuffle

        train_loader = self._get_dataloader(x_train, y_train, bin_train)
        valid_loader = self._get_dataloader(x_val, y_val, bin_val)

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
            for batch, (data, target_power, target_status) in enumerate(
                    train_loader, 1):
                data = data.unsqueeze(1).cuda()
                target_power = target_power.cuda()
                target_status = target_status.cuda()

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs
                # to the model
                output_status = self.model(data).permute(0, 2, 1)
                # calculate the loss
                loss = self.criterion(output_status, target_status)
                # backward pass: compute gradient of the loss with respect
                # to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # record training loss
                train_losses.append(loss.item())

            ######################
            # validate the model #
            ######################
            self.model.eval()  # prep model for evaluation
            for data, target_power, target_status in valid_loader:
                data = data.unsqueeze(1).cuda()
                target_power = target_power.cuda()
                target_status = target_status.cuda()

                # forward pass: compute predicted outputs by passing inputs
                # to the model
                output_status = self.model(data).permute(0, 2, 1)
                # calculate the loss
                loss = self.criterion(output_status, target_status)
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
                    f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                    f'train_loss: {train_loss:.5f} ' +
                    f'valid_loss: {valid_loss:.5f} ')

            print(print_msg)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            if valid_loss < min_loss:
                min_loss = valid_loss
            else:
                loss_up += 1

            if loss_up >= patience:
                break

    def predict(self, x_test):
        tensor_x = torch.Tensor(x_test)
        output_status = self.model(tensor_x).permute(0, 2, 1)
        return output_status
