import os
import subprocess
import sys

import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


# Print versions of used packages.
print('Package Versions')
print('----------------')
print('Pandas version: {}'.format(pd.__version__))
print('Numpy version: {}'.format(np.__version__))
print('Matplotlib version: {}'.format(matplotlib.__version__))


def plot_exp_mod_sim(data_set, title, x_label, y_label,
                     data_x_label=None, data_y_label=None,
                     exp_df=None, model_dfs=None,
                     exp_x_label=None, exp_y_label=None,
                     mod_x_label=None, mod_y_label=None,
                     x_min=0, x_max=600, y_min=0, y_max=700,
                     exp_y_fac=1, mod_y_fac=1, data_y_fac=1,
                     plot_file_name=None, fig_x=9, fig_y=8,
                     exp_color=['b'], mod_color=['k', 'r', 'g']):
    """

    :param data_set:
    :param title:
    :param x_label:
    :param y_label:
    :param data_x_label:
    :param data_y_label:
    :param exp_df:
    :param model_dfs:
    :param exp_x_label:
    :param exp_y_label:
    :param mod_x_label:
    :param mod_y_label:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param exp_y_fac:
    :param mod_y_fac:
    :param data_y_fac:
    :param plot_file_name:
    :param fig_x:
    :param fig_y:
    :param exp_color:
    :param mod_color:
    :return:
    """

    plt.figure(figsize=(fig_x, fig_y))

    # Plot experimental data.
    if exp_df is not None:
        plt.plot(exp_df[[exp_x_label]],
                 exp_df[[exp_y_label]] * exp_y_fac,
                 color=exp_color[0],
                 marker='.', linestyle='None')

    # Plot model data, related to above experiment.
    if model_dfs is not None:
        for i, model_df in enumerate(model_dfs):
            plt.plot(model_df[[mod_x_label]],
                     model_df[[mod_y_label]] * mod_y_fac,
                     color=mod_color[i])

    # Scale color cycle by number of data sets to plot.
    num_plots = len(data_set)
    plt.gca().set_prop_cycle(plt.cycler('color',
                                        plt.cm.viridis(np.linspace(0, 1,
                                                                   num_plots))))

    # Plot data sets, following the coloring definition above.
    for data_series in data_set:
        plt.plot(data_series[[data_x_label]],
                 data_series[[data_y_label]] * data_y_fac)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid()

    if plot_file_name is not None:
        plt.savefig(plot_file_name + '.png')


def normaliser(x, xmin, xmax):
    """
    :param x: Value to be normalised between the provided limits.
    :param xmin: Lower limit.
    :param xmax: Upper limit.
    :return: Normalised value.
    """

    normal = (x-xmin)/(xmax-xmin)
    return normal
