import os
import subprocess
import sys
import pickle

import pandas as pd
import numpy as np
from scipy import constants as const


# Janssens' Procedure
# As described in e.g. the Ignition Handbook, chapter 7, p.260, by Babrauskas
#  (p. 261, top right).

# Calculate transformed time.
def transform_t(time, power=-0.55):
    """
    Transforms the time, following Janssens' procedure as described in
    the Ignition Handbook, chapter 7, p.260, by Babrauskas.

    :param time: numpy array
    :param power: transformation factor, default is -0.55 as described in
                  the Ignition Handbook.
    return: transformed time value
    """

    transformed_time = np.power(time, power)
    return transformed_time


def B_ign(slope):
    """
    From Janssens' procedure description in
    Ignition Handbook p. 261, top right, step 4.
    """
    B_ig = 1 / slope
    return B_ig


def Janssens_T_ign(q_ext,
                   h_coeff,
                   alpha_s=0.88,
                   lower_guess=100,
                   upper_guess=900,
                   step_guess=0.01,
                   collection_threshold=1e-4,
                   Stefan_Boltzmann_const=const.Stefan_Boltzmann / 1000,
                   # from `scipy.constants`
                   T_infty=20 + 273):
    """
    Implementation of solving the equation for the ignition temperature
    provided in Janssens' procedure, as discussed in
    'Ignition Handbook, V. Babrauskas, p. 262'

    :param q_ext: critical heat flux (q_crit),
                  or minimal heat flux (q_min) where ignition was detected
    :param h_coeff: heat transfer coefficient
    :param alpha_s: Taken from Ignition Handbook, as recommended value.
    :param lower_guess: lower limit for the iterative process
    :param upper_guess: upper limit for the iterative process
    :param step_guess: step size for the iterative process
    :param collection_threshold: threshold value to collect possible solutions
    :param Stefan_Boltzmann_const: Stefan Boltzmann constant ~5.67E-11 W/(m² K^4)
    :param T_infty: room temperature in K
    """

    # Initialise list to collect the deltas.
    Janssens_deltas = list()

    # Solve equation by trail-and-error.
    for T_guess in np.arange(lower_guess, upper_guess, step_guess):
        # Left side of the equation.
        left_side = q_ext / (T_guess - T_infty)

        # Right side of the equation.
        right_side = (h_coeff / alpha_s) + \
                     ((Stefan_Boltzmann_const * (T_guess ** 4 - T_infty ** 4)) / \
                      (T_guess - T_infty))

        # Calculate the difference (delta).
        delta = abs(left_side - right_side)

        # Collect values below the threshold.
        if delta < collection_threshold:
            Janssens_deltas.append([delta, T_guess])
    #             print(delta, left_side, right_side, T_guess)

    # Check if something was collected.
    if len(Janssens_deltas) == 0:
        print("* No solution found within the frame of the threshold.")

        # Stop the function here.
        return

        # Find minimum delta.
    Janssens_delta_min = min(Janssens_deltas)
    print("Janssens' delta: {}".format(Janssens_delta_min[0]))
    print("Janssens' T_guess: {} K".format(Janssens_delta_min[1]))

    # Give the guessed temperature back.
    return Janssens_delta_min[1]


def solve_h_eff(alpha_s, q_crit, T_ig, T_infty):

    h_eff = alpha_s * q_crit / (T_ig - T_infty)
    return h_eff


def k_rho_c(h_eff, B_ig, q_crit):
    krc = h_eff**2 * (B_ig/(0.73*q_crit))**(1.828)
    return krc


# hrrpua_surf_nucleus = "ID = '{}',\n      COLOR = '{}',\n      SPEC_ID(1)='{}',\n      MATL_ID(1) = '{}',\n      HRRPUA = {},\n      IGNITION_TEMPERATURE = {},\n      RAMP_Q = '{}',\n      THICKNESS = {} /"
matl_nucleus = "&MATL ID = '{}',\n      CONDUCTIVITY = {},\n      SPECIFIC_HEAT = {},\n      DENSITY = {} /"

def create_hrrpua_ramp(surf_id, hrrpua, time, function_value, ramp_id=None,
                       offset=None, spec_id='PROPANE', color='RASPBERRY',
                       ignition_temp=None, matl_id=None, thickness=None):
    # Nuclei for the input lines.
    hrrpua_surf_nucleus = "&SURF ID = '{}',\n      COLOR = '{}',\n      SPEC_ID(1) ='{}',\n      HRRPUA = {},\n      RAMP_Q = '{}'{} /"
    ramp_nucleus = "&RAMP ID='{}', T={}, F={}/"
    ignition_nucleus = ",\n      IGNITION_TEMPERATURE = {}"
    matl_id__nucleus = ",\n      MATL_ID = '{}'"
    thickness_nucleus = ",\n      THICKNESS = {}"

    # Initialise list to store the input lines.
    input_lines = list()

    # Initialise the optional parameters.
    optional_para = ''

    # Check if offset value is provided. If not, create an offest value of 0.
    if offset is None:
        offset_calc = 0
    else:
        offset_calc = offset

    # Check if ignition temperature was provided.
    if ignition_temp is None:
        ignition_temp = ''
    else:
        optional_para += ignition_nucleus.format(ignition_temp)

    # Check if MATL_ID was provided.
    if matl_id is None:
        matl_id = ''
    else:
        optional_para += matl_id__nucleus.format(matl_id)

    # Check if thickness was provided.
    if thickness is None:
        thickness = ''
    else:
        optional_para += thickness_nucleus.format(thickness)

    # Check if RAMP_ID was provided.
    if ramp_id is None:
        ramp_id = 'BURNER_RAMP'

    # Build SURF input line.
    hrrpua_surf = hrrpua_surf_nucleus.format(surf_id,
                                             color,
                                             spec_id,
                                             hrrpua,
                                             ramp_id,
                                             optional_para)
    input_lines.append(hrrpua_surf)
    input_lines.append('')
    input_lines.append('')
    input_lines.append('')

    # Build the RAMP input lines.
    for id_time, time_val in enumerate(time[offset:]):
        f_val = function_value[id_time + offset_calc] / hrrpua
        ramp_line = ramp_nucleus.format(ramp_id, time_val, f_val)
        input_lines.append(ramp_line)

    # Return the list of input lines.
    return input_lines


