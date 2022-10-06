from cust_libs.sims import simple_ramp, simple_sim, add_eta, plot_coatto, simulation
from cust_libs.data_processing import compute
from cust_libs.modeling import Load_ML_Model
from cust_libs.misc import transf_fun
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import sys

os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')


low_p1 = 0.47
t_lowp1 = 9 * 60 - 10
high_p1 = 0.9
t_highp1 = 4 * 60 - 10
low_p2 = 0.47
t_lowp2 = 5 * 60 - 10
high_p2 = 0.9
t_highp2 = 5 * 60 - 10


t_ramp1 = 20
t_ramp2 = 20
t_ramp3 = 20
t_first_cicle = t_lowp1 + t_ramp1 + t_highp1 + t_ramp2 - 1

p_nom = 410

poi1 = [1, t_lowp1, t_lowp1 + t_ramp1, t_lowp1 + t_ramp1 + t_highp1 -10]
poi2 = [t_first_cicle - 10, t_first_cicle + t_lowp2, t_first_cicle + t_lowp2 + t_ramp3, t_first_cicle + t_lowp2 + t_ramp3
        + t_highp2]

poi = [poi1, poi2]
bess_size = 200

regr_type = 'mono_mod'
method = 'brute-force'


low_p_list1 = np.linspace(low_p1, low_p1, t_lowp1, endpoint=True)
ramp_list1 = np.linspace(low_p1, high_p1, t_ramp1, endpoint=True)
high_p_list1 = np.linspace(high_p1, high_p1, t_highp1, endpoint=True)
ramp_list2 = np.linspace(high_p1, low_p2, t_ramp2, endpoint=True)
low_p_list2 = np.linspace(low_p2, low_p2, t_lowp2, endpoint=True)
ramp_list3 = np.linspace(low_p2, high_p2, t_ramp3, endpoint=True)
high_p_list2 = np.linspace(high_p2, high_p2, t_highp2, endpoint=True)


PwrTOT_rel = []
PwrTOT_rel.extend(low_p_list1)
PwrTOT_rel.extend(ramp_list1)
PwrTOT_rel.extend(high_p_list1)
PwrTOT_rel.extend(ramp_list2)
PwrTOT_rel.extend(low_p_list2)
PwrTOT_rel.extend(ramp_list3)
PwrTOT_rel.extend(high_p_list2)

grad = np.gradient(PwrTOT_rel)
data = pd.DataFrame()
data['PwrTOT_rel'] = PwrTOT_rel
data['Grad_PwrTOT_rel'] = grad
data['PwrTOT'] = data['PwrTOT_rel'] * p_nom

ramp_carico = data
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
plot_coatto(ramp_BESS, ramp_carico, eta - eta_std, bess_size)
