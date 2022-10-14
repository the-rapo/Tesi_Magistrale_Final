import pandas as pd

from cust_libs.sims import simple_ramp, simple_sim, add_eta, plot_coatto, simulation
from cust_libs.data_processing import compute
from cust_libs.modeling import Load_ML_Model
from cust_libs.misc import transf_fun
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')

t_lowp = 6 * 60 - 20
t_highp = 12 * 60 - 20
t_ramp = 40
low_p = 0.47
high_p = 0.85
bess_size = 400

poi = [
    [1200, 1630, 1700, 2730],
    [2800, 3200, 3250, 4000]
    ]

regr_type = 'mono2'

ramp_carico = pd.read_csv('data/processed/Corsini2021/acc/Corsini2021_Acc01.csv')


ramp_BESS = simulation(ramp_carico, poi, bess_size)
ramp_BESS = add_eta(ramp_BESS, regr_type)
ramp_carico = add_eta(ramp_carico, regr_type)

eta_std = compute(ramp_carico, 'avg_eta')
eta = compute(ramp_BESS, 'avg_eta')
plot_coatto(ramp_BESS, ramp_carico, eta - eta_std, bess_size)