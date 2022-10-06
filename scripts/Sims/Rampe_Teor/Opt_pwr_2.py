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
t_highp = 18 * 60 - 20
t_ramp = 40
low_p = 0.48
high_p = 0.87
bess_size = 200
p_nom = 410
poi = [1, t_lowp, t_lowp + t_ramp, t_lowp + t_ramp + t_highp - 1]

regr_type = 'paper'

ramp_carico = simple_ramp(low_p, high_p, t_lowp, t_highp, t_ramp)
ramp_carico = add_eta(ramp_carico, regr_type)

pwrbess_lp = (bess_size * .95 / (t_lowp / 60)) / 410
pwrbess_hp = (bess_size * .95 / (t_highp / 60)) / 410

ramp_BESS = simulation(ramp_carico, poi, bess_size)
ramp_BESS = add_eta(ramp_BESS, regr_type)

eta_std = compute(ramp_carico, 'avg_eta')
eta = compute(ramp_BESS, 'avg_eta')
plot_coatto(ramp_BESS, ramp_carico, eta - eta_std, 200)

rel2tot, tot2rel = transf_fun(p_nom)

fig, ax = plt.subplots(figsize=(18, 9))

ax.plot(ramp_carico['PwrTOT_rel'], label='Carico', linewidth=2)
locs_y = ax.get_yticks()
ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
ax.set_ylabel('Potenza Relativa [%]')
ax.set_xlabel('Tempo [min]')
secax_y = ax.secondary_yaxis('right', functions=(rel2tot, tot2rel))
secax_y.set_ylabel(r'Potenza $[MW]$')

plt.savefig('IO/Opt_Pwr_base', bbox_inches='tight')