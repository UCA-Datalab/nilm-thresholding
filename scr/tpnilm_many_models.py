import collections
import numpy as np
import os
import sys

path_main = os.path.realpath(__file__)
path_main = path_main.rsplit('/', 2)[0]
sys.path.insert(0, path_main)

from better_nilm.ukdale.ukdale_data import load_dataloaders
from better_nilm.model.architecture.tpnilm import PTPNetModel

from better_nilm.model.preprocessing import get_status

from better_nilm.model.scores import classification_scores_dict
from better_nilm.model.scores import regression_scores_dict

from better_nilm.plot_utils import plot_informative_sample
from better_nilm.plot_utils import plot_real_vs_prediction

path_h5 = os.path.join(path_main, 'data/ukdale.h5')
path_data = os.path.join(path_main, '../nilm/data/ukdale')

buildings = [1, 2, 5]
build_id_train = [1, 2, 5]
build_id_valid = [1]
build_id_test = [1]
appliances = ['fridge', 'dish_washer', 'washing_machine']

activation_w = 1
power_w = 1

dates = {1: ('2013-04-12', '2014-12-15'),
         2: ('2013-05-22', '2013-10-03 6:16'),
         5: ('2014-04-29', '2014-09-01')}

train_size = 0.8
valid_size = 0.1

seq_len = 480
border = 16
period = '1min'
power_scale = 2000.
num_appliances = len(appliances)

batch_size = 32
learning_rate = 1.E-4
dropout = 0.1
epochs = 300
patience = 300

num_models = 5

# Load data

params = load_dataloaders(path_h5, path_data, buildings, appliances,
                          build_id_train=build_id_train,
                          build_id_valid=build_id_valid,
                          build_id_test=build_id_test,
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

    model = PTPNetModel(seq_len=seq_len, border=border,
                        out_channels=num_appliances,
                        init_features=32,
                        learning_rate=learning_rate, dropout=dropout,
                        activation_w=activation_w, power_w=power_w)

    model.train_with_dataloader(dl_train, dl_valid,
                                epochs=epochs,
                                patience=patience)

    # Test

    x_true, p_true, s_true, p_hat, s_hat = model.predict_loader(dl_test)

    # Denormalize power values
    p_true = np.multiply(p_true, power_scale)
    p_hat = np.multiply(p_hat, power_scale)
    p_hat[p_hat < 0.] = 0.

    # Activation scores

    s_hat[s_hat >= .5] = 1
    s_hat[s_hat < 0.5] = 0

    # Get power values from status
    sp_hat = np.multiply(np.ones(s_hat.shape), means[:, 0])
    sp_on = np.multiply(np.ones(s_hat.shape), means[:, 1])
    sp_hat[s_hat == 1] = sp_on[s_hat == 1]

    class_scores = classification_scores_dict(s_hat, s_true, appliances)
    reg_scores = regression_scores_dict(sp_hat, p_true, appliances)
    act_scores += [class_scores, reg_scores]

    print('Activation scores')
    print(class_scores)
    print(reg_scores)

    # Power scores

    # Get status from power values
    ps_hat = get_status(p_hat, thresholds)

    class_scores = classification_scores_dict(ps_hat, s_true, appliances)
    reg_scores = regression_scores_dict(p_hat, p_true, appliances)
    pow_scores += [class_scores, reg_scores]

    print('Power scores')
    print(class_scores)
    print(reg_scores)

# List scores

scores = {'activation': {},
          'power': {}}
for app in appliances:
    counter = collections.Counter()
    for sc in act_scores:
        counter.update(sc[app])
    scores['activation'][app] = {k: round(v, 6) / num_models for k, v in
                                 dict(counter).items()}

    counter = collections.Counter()
    for sc in pow_scores:
        counter.update(sc[app])
    scores['power'][app] = {k: round(v, 6) / num_models for k, v in
                            dict(counter).items()}

# Plot

path_plots = os.path.join(path_main, 'outputs')
if not os.path.isdir(path_plots):
    os.mkdir(path_plots)

path_plots = os.path.join(path_plots, 'tpnilm')
if not os.path.isdir(path_plots):
    os.mkdir(path_plots)

name = f"seq_{str(seq_len)}_{period}_aw_{str(activation_w)}_pw_{str(power_w)}"
path_plots = os.path.join(path_plots, name)
if not os.path.isdir(path_plots):
    os.mkdir(path_plots)

path_scores = os.path.join(path_plots, 'scores.txt')
with open(path_scores, "w") as text_file:
    for key, dic1 in scores.items():
        text_file.write(f"{key}\n------------------------------------------\n")
        for app, dic2 in dic1.items():
            text_file.write(f"{app} \n")
            for name, value in dic2.items():
                text_file.write(f"{name}: {value}\n")
            text_file.write('----------------------------------------------\n')
        text_file.write('==================================================\n')

# Compute period of x axis
if period.endswith('min'):
    period_x = int(period.replace('min', ''))
elif period.endswith('s'):
    period_x = float(period.replace('s', '')) / 60

for idx, app in enumerate(appliances):
    savefig = os.path.join(path_plots, f"{app}_classification.png")
    plot_informative_sample(p_true, s_true, sp_hat, s_hat, records=seq_len,
                            app_idx=idx, scale=1., period=period_x, dpi=180,
                            savefig=savefig)

    savefig = os.path.join(path_plots, f"{app}_regression.png")
    plot_informative_sample(p_true, s_true, p_hat, ps_hat, records=seq_len,
                            app_idx=idx, scale=1., period=period_x, dpi=180,
                            savefig=savefig)
