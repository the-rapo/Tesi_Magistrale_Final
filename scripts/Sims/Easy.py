import pandas as pd

from cust_libs.sims import simple_ramp, simple_sim, add_eta, plot_coatto
from cust_libs.data_processing import compute
from cust_libs.modeling import Load_ML_Model
from cust_libs.misc import transf_fun
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')

t_lowp = 2 * 60
t_highp = 2 * 60
t_ramp = 20
low_p = 0.47
high_p = 0.85

LP_tresh = 0.64
HP_tresh = 0.72

bess_size = 200
method = 'manual'
regr_type = 'mono_mod'
acc_path = 'data/processed/Corsini2021/acc/Corsini2021_Acc04.csv'
acc = pd.read_csv(acc_path)
acc_corr = add_eta(acc, regr_type)
eta_std = compute(acc_corr, 'avg_eta')
if method == 'auto':
    rendimento_max = 0
    winner_LP = 0
    winner_HP = 0
    for i in np.arange(0.45, 0.65, 0.01):
        for j in np.arange(i + 0.01, 0.9, 0.01):
            ramp_BESS = simple_sim(i, j, bess_size, acc_corr)
            ramp_BESS = add_eta(ramp_BESS, regr_type)
            eta = compute(ramp_BESS, 'avg_eta')
            if eta > rendimento_max:
                rendimento_max = eta
                winner_LP = i
                winner_HP = j

    print('Migliori parametri')
    print('L_op' + str(winner_LP))
    print('H_op' + str(winner_HP))

    ramp_BESS = simple_sim(winner_LP, winner_HP, bess_size, acc_corr)
    ramp_BESS = add_eta(ramp_BESS, regr_type)
    eta = compute(ramp_BESS, 'avg_eta')
    plot_coatto(ramp_BESS, acc_corr, eta - eta_std)

if method == 'manual':
    ramp_BESS = simple_sim(LP_tresh, HP_tresh, bess_size, acc_corr)
    ramp_BESS = add_eta(ramp_BESS, regr_type)
    eta = compute(ramp_BESS, 'avg_eta')
    plot_coatto(ramp_BESS, acc_corr, eta - eta_std)

# 1 - 0.55 | 0.79