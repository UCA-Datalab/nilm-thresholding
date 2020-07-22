import collections
import numpy as np
import os

from better_nilm.model.preprocessing import get_status

from better_nilm.model.scores import classification_scores_dict
from better_nilm.model.scores import regression_scores_dict

from better_nilm.plot_utils import plot_informative_sample


def process_model_outputs(p_true, p_hat, s_hat,
                          power_scale, means, thresholds):
    # Denormalize power values
    p_true = np.multiply(p_true, power_scale)
    p_hat = np.multiply(p_hat, power_scale)
    p_hat[p_hat < 0.] = 0.

    s_hat[s_hat >= .5] = 1
    s_hat[s_hat < 0.5] = 0

    # Get power values from status
    sp_hat = np.multiply(np.ones(s_hat.shape), means[:, 0])
    sp_on = np.multiply(np.ones(s_hat.shape), means[:, 1])
    sp_hat[s_hat == 1] = sp_on[s_hat == 1]

    # Get status from power values
    ps_hat = get_status(p_hat, thresholds)

    return p_true, p_hat, s_hat, sp_hat, ps_hat


def get_model_scores(model, dl_test, power_scale, means, thresholds):
    """
    Trains and test a model. Returns its activation and power scores.
    """

    # Test
    x_true, p_true, s_true, p_hat, s_hat = model.predict_loader(dl_test)

    p_true, p_hat, s_hat, \
    sp_hat, ps_hat = process_model_outputs(p_true, p_hat, s_hat,
                                           power_scale, means, thresholds)

    # classification scores

    class_scores = classification_scores_dict(s_hat, s_true, appliances)
    reg_scores = regression_scores_dict(sp_hat, p_true, appliances)
    act_scores = [class_scores, reg_scores]

    print('classification scores')
    print(class_scores)
    print(reg_scores)

    # regression scores

    class_scores = classification_scores_dict(ps_hat, s_true, appliances)
    reg_scores = regression_scores_dict(p_hat, p_true, appliances)
    pow_scores = [class_scores, reg_scores]

    print('regression scores')
    print(class_scores)
    print(reg_scores)

    return act_scores, pow_scores


def list_scores(appliances, act_scores, pow_scores, num_models):
    """
    List scores in dictionary format.
    """
    scores = {'classification': {},
              'regression': {}}

    for app in appliances:
        counter = collections.Counter()
        for sc in act_scores:
            counter.update(sc[app])
        scores['classification'][app] = {k: round(v, 6) / num_models for k, v
                                         in
                                         dict(counter).items()}

        counter = collections.Counter()
        for sc in pow_scores:
            counter.update(sc[app])
        scores['regression'][app] = {k: round(v, 6) / num_models for k, v in
                                     dict(counter).items()}

    return scores


def plot_store_results(path_main, model_name, seq_len, period, class_w, reg_w,
                       threshold_method, train_size, valid_size, num_models,
                       batch_size, learning_rate, dropout, epochs, patience,
                       scores, appliances,
                       model, dl_test, power_scale, means, thresholds):

    path_plots = os.path.join(path_main, 'outputs')
    if not os.path.isdir(path_plots):
        os.mkdir(path_plots)

    path_plots = os.path.join(path_plots, model_name)
    if not os.path.isdir(path_plots):
        os.mkdir(path_plots)

    name = f"seq_{str(seq_len)}_{period}_clas_{str(class_w)}" \
           f"_reg_{str(reg_w)}_{threshold_method}"
    path_plots = os.path.join(path_plots, name)
    if not os.path.isdir(path_plots):
        os.mkdir(path_plots)

    path_scores = os.path.join(path_plots, 'scores.txt')

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

    # Model values

    x_true, p_true, s_true, p_hat, s_hat = model.predict_loader(dl_test)

    p_true, p_hat, s_hat, \
    sp_hat, ps_hat = process_model_outputs(p_true, p_hat, s_hat,
                                           power_scale, means, thresholds)

    for idx, app in enumerate(appliances):
        savefig = os.path.join(path_plots, f"{app}_classification.png")
        plot_informative_sample(p_true, s_true, sp_hat, s_hat,
                                records=seq_len,
                                app_idx=idx, scale=1., period=period_x,
                                dpi=180,
                                savefig=savefig)

        savefig = os.path.join(path_plots, f"{app}_regression.png")
        plot_informative_sample(p_true, s_true, p_hat, ps_hat,
                                records=seq_len,
                                app_idx=idx, scale=1., period=period_x,
                                dpi=180,
                                savefig=savefig)
