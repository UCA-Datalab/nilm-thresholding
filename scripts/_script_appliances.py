import collections
import numpy as np
import os

from better_nilm.ukdale.ukdale_data import load_dataloaders

from better_nilm.model.preprocessing import get_status

from better_nilm.model.scores import classification_scores_dict
from better_nilm.model.scores import regression_scores_dict

from better_nilm.plot_utils import plot_informative_sample

# Set default variables

path_h5 = None
path_data = None
path_main = None
buildings = None
appliances = None
class_w = 0
reg_w = 0
dates = None
train_size = 0
valid_size = 0
seq_len = 480
border = 16
period = '1min'
power_scale = 2000.
batch_size = 32
learning_rate = 0
dropout = 0
epochs = 0
patience = 0
num_models = 0
num_appliances = 0
model_name = None
model_params = None

# Import from other scripts

from __main__ import *

# Set output path

path_plots = os.path.join(path_main, 'outputs')
if not os.path.isdir(path_plots):
    os.mkdir(path_plots)

path_plots = os.path.join(path_plots, model_name)
if not os.path.isdir(path_plots):
    os.mkdir(path_plots)

# Load data

for building in buildings:
    for appliance in appliances:

        print(f"\nHouse {building}, Appliance {appliance}\n")

        params = load_dataloaders(path_h5, path_data, building, appliance,
                                  dates=dates, period=period,
                                  train_size=train_size, valid_size=valid_size,
                                  batch_size=batch_size, seq_len=seq_len,
                                  border=border, power_scale=power_scale,
                                  return_kmeans=True)

        dl_train, dl_valid, dl_test, kmeans = params
        thresholds, means = kmeans

        # Training
        act_scores = []
        pow_scores = []

        for i in range(num_models):
            print(f"\nModel {i + 1}\n")

            model = eval(model_name)(**model_params)

            model.train_with_dataloader(dl_train, dl_valid,
                                        epochs=epochs,
                                        patience=patience)

            # Test

            x_true, p_true, s_true, \
            p_hat, s_hat = model.predict_loader(dl_test)

            # Denormalize power values
            p_true = np.multiply(p_true, power_scale)
            p_hat = np.multiply(p_hat, power_scale)
            p_hat[p_hat < 0.] = 0.

            # classification scores

            s_hat[s_hat >= .5] = 1
            s_hat[s_hat < 0.5] = 0

            # Get power values from status
            sp_hat = np.multiply(np.ones(s_hat.shape), means[:, 0])
            sp_on = np.multiply(np.ones(s_hat.shape), means[:, 1])
            sp_hat[s_hat == 1] = sp_on[s_hat == 1]

            class_scores = classification_scores_dict(s_hat, s_true,
                                                      appliance)
            reg_scores = regression_scores_dict(sp_hat, p_true, appliance)
            act_scores += [class_scores, reg_scores]

            print('classification scores')
            print(class_scores)
            print(reg_scores)

            # regression scores

            # Get status from power values
            ps_hat = get_status(p_hat, thresholds)

            class_scores = classification_scores_dict(ps_hat, s_true,
                                                      appliance)
            reg_scores = regression_scores_dict(p_hat, p_true, appliance)
            pow_scores += [class_scores, reg_scores]

            print('regression scores')
            print(class_scores)
            print(reg_scores)

        # List scores

        scores = {'classification': {},
                  'regression': {}}
        counter = collections.Counter()
        for sc in act_scores:
            counter.update(sc[appliance])
        scores['classification'][appliance] = {k: round(v, 6) / num_models for
                                               k, v in dict(counter).items()}

        counter = collections.Counter()
        for sc in pow_scores:
            counter.update(sc[appliance])
        scores['regression'][appliance] = {k: round(v, 6) / num_models for k, v
                                           in dict(counter).items()}

        # Plot
        path_app = os.path.join(path_plots, f"house_{building}_{appliance}")
        if not os.path.isdir(path_app):
            os.mkdir(path_app)

        name = f"seq_{str(seq_len)}_{period}_clas_{str(class_w)}" \
               f"_reg_{str(reg_w)}"
        path_app = os.path.join(path_app, name)
        if not os.path.isdir(path_app):
            os.mkdir(path_app)

        path_scores = os.path.join(path_app, 'scores.txt')
        with open(path_scores, "w") as text_file:
            text_file.write(f"Train size: {train_size}\n"
                            f"Validation size: {valid_size}\n"
                            f"Number of models: {num_models}\n"
                            f"Batch size: {batch_size}\n"
                            f"Learning rate: {learning_rate}\n"
                            f"Dropout: {dropout}\n"
                            f"Epochs: {epochs}\n"
                            f"Patience: {patience}\n"
                            f"=============================================\n")
            for key, dic1 in scores.items():
                text_file.write(
                    f"{key}\n------------------------------------------\n")
                for app, dic2 in dic1.items():
                    text_file.write(f"{app} \n")
                    for name, value in dic2.items():
                        text_file.write(f"{name}: {value}\n")
                    text_file.write(
                        '----------------------------------------------\n')
                text_file.write(
                    '==================================================\n')

        # Compute period of x axis
        if period.endswith('min'):
            period_x = int(period.replace('min', ''))
        elif period.endswith('s'):
            period_x = float(period.replace('s', '')) / 60

        savefig = os.path.join(path_app, "classification.png")
        plot_informative_sample(p_true, s_true, sp_hat, s_hat,
                                records=seq_len,
                                app_idx=0, scale=1., period=period_x,
                                dpi=180,
                                savefig=savefig)

        savefig = os.path.join(path_app, "regression.png")
        plot_informative_sample(p_true, s_true, p_hat, ps_hat,
                                records=seq_len,
                                app_idx=0, scale=1., period=period_x,
                                dpi=180,
                                savefig=savefig)
