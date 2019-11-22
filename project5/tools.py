# howdy
# ######### IMPORTS #############
import time
import numpy as np
from numba import jit, prange
from random import random, seed
import matplotlib.pyplot as plt
# ######### IMPORTS #############

# ########## GENERAL SHAPE PLOT ############################
shape_plot = False
if shape_plot == True:
    steps = np.linspace(0,99,101)
    vals = np.random.randint(0,99, size = steps.shape[0])
    beta = 1 / np.mean(vals)
    vals = vals[np.argsort(vals)]
    omega = beta*np.exp(-beta*steps)
    norm_steps = steps/ np.mean(vals)

    fig = plt.figure()
    plt.plot(norm_steps, omega)
    plt.show()
# ########## GENERAL SHAPE PLOT ############################

class FinanceExperiment:
    def __init__(
            self,
            agents = 500, starting_amt = 1000,
            MCsteps = 1e3, transactions = 1e7,
            lam = 0, alp = 0, gam = 0,
            bin_max = 'BRUTE',
            filename = None
        ):

        if (type(filename) is not str):
            raise TypeError('Put in a file name or else...')
        if (bin_max not in ['BRUTE', 'ELEGANT']):
            raise TypeError("bin_max must be in ['BRUTE','ELEGANT']")
        # INIT ARGUEMENTS
        self._agents = agents
        self._starting_amt = starting_amt
        self._MCsteps = int(MCsteps)
        self._transactions = int(transactions)
        self._lam = lam
        self._alp = alp
        self._gam = gam
        self._filename = filename
        self._equity = np.zeros(agents)
        self._diff_vec = np.zeros(int(MCsteps)-1)

        # BIN ARGS
        self._bin_size = 0.01 * starting_amt
        if (bin_max == 'BRUTE'):
            self._bin_max = 10 * starting_amt
        elif (bin_max == 'ELEGANT'):
            self._bin_max = 2*starting_amt/np.sqrt(gam + 0.01)
        self._bin_num = int(self._bin_max/self._bin_size)
        self._bin_steps = np.linspace(0, self._bin_max, self._bin_num)
        self._bin_vals = np.zeros(self._bin_num)

        self._val_holder = np.array([np.zeros(self._bin_num) for step in range(self._MCsteps)])

    # wrap jitted function
    def montecarlo(self):
        _montecarlo(self._MCsteps, self._starting_amt, self._transactions, self._agents, self._equity,
                    self._bin_num, self._bin_size, self._bin_vals, self._val_holder)

    # wrap jitted function
    def error_calc(self):
        _error_calc(self._MCsteps, self._diff_vec, self._val_holder)

    def saveBinParams(self):
        np.savez(self._filename + 'BinParams', binSize = self._bin_size, binMax = self._bin_max,
                 binNumber = self._bin_num, binSteps = self._bin_steps)
    def saveBinVals(self):
        np.savez(self._filename + 'BinVals', final_dist = self._bin_vals,
                 error_vals = self._diff_vec, all_vals = self._val_holder)

# jitted for speed yo
@jit(nopython = True, parallel = True)
def _error_calc(MCsteps, diff_vec, val_holder):
    for i in prange(0, MCsteps-1):
        diff_vec[i]=np.linalg.norm(val_holder[i]/(i+1)-val_holder[i+1]/(i+2))

# jitted function for calculations
@jit(nopython = True, parallel = True)
def _montecarlo(MCsteps, starting_amt, transactions, agents, equity, bin_num, bin_size, bin_vals, val_holder):

    for mc_val in prange(MCsteps):
        print(mc_val)
        equity.fill(starting_amt)
        for deals in range(transactions):
            # TRANSACTION CUZZO
            eps = np.random.uniform(0,1)
            temp = np.random.choice(agents, 2, replace = False)
            idx_i = temp[0]
            idx_j = temp[1]
            m1 = eps * (equity[idx_i] + equity[idx_j])
            m2 = (1 - eps) * (equity[idx_i] + equity[idx_j])
            equity[idx_i] = m1
            equity[idx_j] = m2

        # UPDATE BINS YO
        for i in range(agents):
            for j in range(bin_num):
                if ((equity[i] > j*bin_size) and (equity[i] < (j+1)*bin_size)):
                    bin_vals[j] += 1
        val_holder[mc_val] = bin_vals

if __name__ == '__main__':
    a = FinanceExperiment(agents = 500, starting_amt = 100, MCsteps = 1e3, transactions = 1e4,
                          filename = 'parta_')
    start = time.time()
    a.montecarlo()
    end = time.time()
    print('MC time: ', (end-start)/60)
    a.error_calc()
    a.saveBinParams()
    a.saveBinVals()




















#
