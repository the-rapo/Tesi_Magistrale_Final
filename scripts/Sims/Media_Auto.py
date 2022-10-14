import pandas as pd

from cust_libs.sims import simple_ramp, simple_sim, add_eta, plot_coatto, media
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
regr_type = 'mono2_mod'
out_fold = 'IO/sim_acc'

for i in range(55):
    acc_no = i + 1
    str(acc_no).zfill(2)
    savepath = out_fold + '/' + 'media-' + str(acc_no).zfill(2) + '_' + regr_type + '.png'
    acc_path = 'data/processed/Corsini2021/acc/Corsini2021_Acc' + str(acc_no).zfill(2) + '.csv'
    acc = pd.read_csv(acc_path)
    acc_corr = add_eta(acc, regr_type)
    eta_std = compute(acc_corr, 'avg_eta')
    ramp_BESS = media(acc_corr)
    ramp_BESS = add_eta(ramp_BESS, regr_type)
    eta = compute(ramp_BESS, 'avg_eta')
    plot_coatto(ramp_BESS, acc_corr, eta - eta_std, save=savepath)
    print(str(acc_no).zfill(2) + ' / ' + str(55))
