import os
import subprocess
import sys

# Build gif from single images with ffmpeg.
# ffmpeg -i filename_%04d.png output.gif

# Extract frames from video file.
# ffmpeg -i video.file frame%04d.png

import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp


# Print versions of used packages.
print('Package Versions')
print('----------------')
print('Pandas version: {}'.format(pd.__version__))
print('Numpy version: {}'.format(np.__version__))
print('Matplotlib version: {}'.format(matplotlib.__version__))


def plot_exp_mod_sim(title, x_label, y_label, data_set=None,
                     data_x_label=None, data_y_label=None,
                     exp_df=None, model_dfs=None,
                     exp_x_label=None, exp_y_label=None,
                     mod_x_label=None, mod_y_label=None,
                     x_min=0, x_max=600, y_min=0, y_max=700,
                     exp_y_fac=1, mod_y_fac=1, data_y_fac=1,
                     plot_file_name=None, fig_x=9, fig_y=8, dpi=320,
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
    :param dpi:
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

    if data_set is not None:
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
        plt.savefig(plot_file_name + '.png', dpi=dpi)


def normaliser(x, xmin, xmax):
    """
    Normalises a parameters "location" within a given range. Lower range
    limit is set to 0, upper range limit is set to 1.

    :param x: Value to be normalised between the provided limits.
    :param xmin: Lower limit.
    :param xmax: Upper limit.
    :return: Normalised value.
    """

    normalised_value = (x-xmin)/(xmax-xmin)
    return normalised_value


def reverser(normal_value, xmin, xmax):
    """
    Maps normalised values back to the desired range.

    :param normal_value: Normalised value to be mapped onto the provided range.
    :param xmin: Lower limit.
    :param xmax: Upper limit.
    :return: Reversed value.
    """

    reversed_value = normal_value * (xmax - xmin) + xmin
    return reversed_value


def adjust_range(ranges_dict, new_ranges, percent=True):
    """
    This function aims to adjust sampling ranges as used during the inverse
    modelling process with PROPTI/SPOTPY. Input is a dictionary, where the
    keys are the parameter labels and the values are a list containing the
    lower and the upper limit of the sampling range (`ranges_dict`).
    New limits are to be provided as list of lists (`new_ranges`). Each
    nested list contains a string matching a key of `ranges_dict` and a new
    value. Positive values are assumed to increase the upper, negative values
    the lower limit. The `percent` flag determines if the provided values are
    to adjust the range as percentages of the sampling range (`True`) or if the
    respective values are to be taken as provided (False).
    New limits are appended to the existing list, leading to:
    [original min, original max, new min, new max].
    For parameters that are not to be changed the original limits will
    simply be copied.

    :param ranges_dict: Dictionary with the parameter label as key and a list
        of upper, lower limits.
    :param new_ranges: List of lists, containig parameter label as string and
        new value, possibly as percentage (see the flag).
    :param percent: If `True`, values in `new_ranges` need to be between 0
        and 1. If `False`, values are taken as provided.
    :return: Nothing - the dictionary is adjusted in place (list append).

    """
    # Extract all the keys of the dictionary and
    # store as list.
    key_list = list(ranges_dict.keys())

    for key in key_list:
        for para in new_ranges:
            # Check if key is in list of ranges to be adjusted.
            if para[0] == key:
                if percent is True:
                    new_limit = (ranges_dict[key][1] - ranges_dict[key][
                        0]) * abs(para[1])
                else:
                    new_limit = abs(para[1])

                # If new value is negative, assume range extention
                # at lower end; else extent the upper end.
                if para[1] < 0:
                    new_min = ranges_dict[key][0] - new_limit
                    new_max = ranges_dict[key][1]
                    print(key, "<0: ", ranges_dict[key][0], new_min)
                else:
                    new_min = ranges_dict[key][0]
                    new_max = ranges_dict[key][1] + new_limit
                    print(key, ">=0: ", ranges_dict[key][1], new_max)

                # Stop loop if a match is found.
                break

            # If keys do not match, simply copy the existing ones.
            else:
                new_min = ranges_dict[key][0]
                new_max = ranges_dict[key][1]

        print("* ", key)
        print(ranges_dict[key][0], ranges_dict[key][1])
        print(new_min, new_max)
        print("------")

        # Extent the existing min/max list of the range with
        # the new limit values.
        if len(ranges_dict[key]) == 2:
            ranges_dict[key].append(new_min)
            ranges_dict[key].append(new_max)
        else:
            ranges_dict[key][2] = new_min
            ranges_dict[key][3] = new_max


def run_job2(file_path):
    """
    Provided with a path to a simulation input file, a sub-process
    is created and the simulation conducted.

    :param file_path: path to the simulation input file
        to be executed
    """
    cwd = os.getcwd()
    print(os.path.split(file_path))
    wd, fn = os.path.split(file_path)
    os.chdir(wd)
    subprocess.call("export set OMP_NUM_THREADS=1; \
                    fds {}".format(fn),
                    shell=True)
    os.chdir(cwd)


# if __name__ == "__main__":
#     pool = mp.Pool(processes=2)
#     pool.map(run_job2, input_files[:6])
