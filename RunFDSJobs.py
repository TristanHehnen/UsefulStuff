import os
import subprocess

import multiprocessing as mp


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


input_files = ['SimpleConeLHS/C219_13b_lhs_0000/C219_13b_lhs_0000.fds',
               'SimpleConeLHS/C219_13b_lhs_0001/C219_13b_lhs_0001.fds',
               'SimpleConeLHS/C219_13b_lhs_0002/C219_13b_lhs_0002.fds',
               'SimpleConeLHS/C219_13b_lhs_0003/C219_13b_lhs_0003.fds',
               'SimpleConeLHS/C219_13b_lhs_0004/C219_13b_lhs_0004.fds',
               'SimpleConeLHS/C219_13b_lhs_0005/C219_13b_lhs_0005.fds']


if __name__ == "__main__":
    pool = mp.Pool(processes=2)
    pool.map(run_job2, input_files[:])
