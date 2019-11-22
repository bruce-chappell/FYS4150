import numpy as np
from numba import jit, prange
from random import random, seed
import matplotlib.pyplot as plt
import time
from itertools import *

@jit(nopython = True, parallel = True)
def montecarlo(agents = 100, MCsteps = 1e4, starting_amt = 100, transactions = 1e4, filename = 'def',
               calc_diff = False, write_file = True):

    equity = np.zeros(agents)
    bin_size = 0.01 * starting_amt
    # bin_max = 2 * starting_amt / np.sqrt(0.01)
    bin_max = 10*starting_amt
    bin_num = int(bin_max / bin_size)
    bin_vals = np.zeros(bin_num)
    bin_steps = np.linspace(0, bin_max, bin_num)
    diff_vec = np.zeros(int(MCsteps)+1)



    for mc_val in prange(int(MCsteps)):

        equity.fill(starting_amt)
        for deals in range(int(transactions)):
            # TRANSACTION CUZZO
            eps = np.random.uniform(0,1)
            temp = np.random.choice(agents, 2, replace = False)
            idx_i = temp[0]
            idx_j = temp[1]
            m1 = eps * (equity[idx_i] + equity[idx_j])
            m2 = (1 - eps) * (equity[idx_i] + equity[idx_j])
            equity[idx_i] = m1
            equity[idx_j] = m2

        temp_b = np.copy(bin_vals)

        # UPDATE BINS YO
        for i in range(agents):
            for j in range(bin_num):
                if ((equity[i] > j*bin_size) and (equity[i] < (j+1)*bin_size)):
                    bin_vals[j] += 1

        # DISTRIBUTION DIFFERNCES B
        if (calc_diff == True):
            if (mc_val >= 1):
                diff_vec[mc_val]=np.linalg.norm(temp_b/(mc_val)-bin_vals/(mc_val+1))

    return bin_vals, bin_steps, equity, diff_vec

@jit(nopython = True)
def slow_montecarlo(agents = 100, MCsteps = 1e4, starting_amt = 100, transactions = 1e4, filename = 'def',
               calc_diff = False, write_file = True):

    equity = np.zeros(agents)
    bin_size = 0.01 * starting_amt
    # bin_max = 2 * starting_amt / np.sqrt(0.01)
    bin_max = 10*starting_amt
    bin_num = int(bin_max / bin_size)
    bin_vals = np.zeros(bin_num)
    bin_steps = np.linspace(0, bin_max, bin_num)
    diff_vec = np.zeros(int(MCsteps)+1)



    for mc_val in range(int(MCsteps)):

        equity.fill(starting_amt)
        for deals in range(int(transactions)):
            # TRANSACTION CUZZO
            eps = np.random.uniform(0,1)
            temp = np.random.choice(agents, 2, replace = False)
            idx_i = temp[0]
            idx_j = temp[1]
            m1 = eps * (equity[idx_i] + equity[idx_j])
            m2 = (1 - eps) * (equity[idx_i] + equity[idx_j])
            equity[idx_i] = m1
            equity[idx_j] = m2

        temp_b = np.copy(bin_vals)

        # UPDATE BINS YO
        for i in range(agents):
            for j in range(bin_num):
                if ((equity[i] > j*bin_size) and (equity[i] < (j+1)*bin_size)):
                    bin_vals[j] += 1

        # DISTRIBUTION DIFFERNCES B
        if (calc_diff == True):
            if (mc_val >= 1):
                diff_vec[mc_val]=np.linalg.norm(temp_b/(mc_val)-bin_vals/(mc_val+1))

    return bin_vals, bin_steps, equity, diff_vec

if __name__ == '__main__':

    b = np.arange(1,6)
    b = b.reshape(5,1)
    print(b)
    c = np.arange(1,6)
    c = b.reshape(5,1)
    d = np.arange(1,6)
    d = b.reshape(5,1)

    f = np.array(list(zip(b,c,d)))
    print(f.reshape(15,1))
