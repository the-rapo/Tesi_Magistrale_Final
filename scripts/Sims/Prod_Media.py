import pandas as pd

from cust_libs.sims import simple_ramp, simple_sim, add_eta, plot_coatto, simulation, media
from cust_libs.data_processing import compute
from cust_libs.modeling import Load_ML_Model
from cust_libs.misc import transf_fun
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')

regr_type = 'mono2_mod'

ramp_carico = pd.read_csv('data/processed/Corsini2021/acc/Corsini2021_Acc04.csv')


ramp_BESS = media(ramp_carico)
ramp_BESS = add_eta(ramp_BESS, regr_type)
ramp_carico = add_eta(ramp_carico, regr_type)

eta_std = compute(ramp_carico, 'avg_eta')
eta = compute(ramp_BESS, 'avg_eta')
plot_coatto(ramp_BESS, ramp_carico, eta - eta_std)
print(max(ramp_BESS['BESS_SOC']))

