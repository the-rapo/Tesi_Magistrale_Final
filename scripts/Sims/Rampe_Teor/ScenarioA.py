from cust_libs.sims import simple_ramp, simple_sim, add_eta, plot_coatto, simulation
from cust_libs.data_processing import compute
from cust_libs.modeling import Load_ML_Model
from cust_libs.misc import transf_fun
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')

t_lowp = 9 * 60 - 20
t_highp = 15 * 60 - 20
t_ramp = 40
low_p = 0.48
high_p = 0.9
bess_size = 200
p_nom = 410
poi = [1, t_lowp, t_lowp + t_ramp, t_lowp + t_ramp + t_highp - 1]

regr_type = 'mono_mod'
method = 'brute-force'


ramp_carico = simple_ramp(low_p, high_p, t_lowp, t_highp, t_ramp)
ramp_carico = add_eta(ramp_carico, regr_type)

if method == 'brute-force':
    rendimento_max = 0
    winner_LP = 0
    winner_HP = 0
    for i in np.arange(0.45, 0.65, 0.01):
        for j in np.arange(i + 0.01, 0.9, 0.01):
            ramp_BESS = simple_sim(i, j, bess_size, ramp_carico)
            ramp_BESS = add_eta(ramp_BESS, regr_type)
            eta = compute(ramp_BESS, 'avg_eta')
            if eta > rendimento_max:
                rendimento_max = eta
                winner_LP = i
                winner_HP = j
        print('Migliori parametri')
        print('L_op' + str(winner_LP))
        print('H_op' + str(winner_HP))

    ramp_BESS = simple_sim(winner_LP, winner_HP, bess_size, ramp_carico)
elif method == 'opt':
    ramp_BESS = simulation(ramp_carico, poi, bess_size)
else:
    sys.exit("Errore")

ramp_BESS = add_eta(ramp_BESS, regr_type)

eta_std = compute(ramp_carico, 'avg_eta')
eta = compute(ramp_BESS, 'avg_eta')
plot_coatto(ramp_BESS, ramp_carico, eta - eta_std, 200)
