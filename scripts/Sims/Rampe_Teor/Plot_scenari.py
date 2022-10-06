from cust_libs.sims import simple_ramp, simple_sim, add_eta, plot_coatto, simulation
from cust_libs.data_processing import compute
from cust_libs.modeling import Load_ML_Model
from cust_libs.misc import transf_fun
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')

t_lowp = 9 * 60 - 20
t_highp = 15 * 60 - 20
t_ramp = 40
low_p = 0.47
high_p = 0.9
bess_size = 200
p_nom = 410

ramp_caricoA = simple_ramp(low_p, high_p, t_lowp, t_highp, t_ramp)

t_lowp = 15 * 60 - 20
t_highp = 6 * 60 - 20
t_ramp = 40
low_p = 0.47
high_p = 0.9

ramp_caricoB = simple_ramp(low_p, high_p, t_lowp, t_highp, t_ramp)

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

ramp_caricoC = data

rel2tot, tot2rel = transf_fun(p_nom)
mpl.rcParams["font.size"] = 18

# Scenario A

fig, ax = plt.subplots(figsize=(18, 9))

ax.plot(ramp_caricoA['PwrTOT_rel'], label='Scenario A', linewidth=3, color='tab:blue')
locs_y = ax.get_yticks()
ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
ax.set_ylabel('Potenza Relativa [%]')
ax.set_xlabel('Tempo [min]')
secax_y = ax.secondary_yaxis('right', functions=(rel2tot, tot2rel))
secax_y.set_ylabel(r'Potenza $[MW]$')

plt.savefig('IO/Opt_Pwr_base_scenarioA', bbox_inches='tight')

# Scenario B

fig, ax = plt.subplots(figsize=(18, 9))

ax.plot(ramp_caricoB['PwrTOT_rel'], label='Scenario A', linewidth=3, color='tab:green')
locs_y = ax.get_yticks()
ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
ax.set_ylabel('Potenza Relativa [%]')
ax.set_xlabel('Tempo [min]')
secax_y = ax.secondary_yaxis('right', functions=(rel2tot, tot2rel))
secax_y.set_ylabel(r'Potenza $[MW]$')

plt.savefig('IO/Opt_Pwr_base_scenarioB', bbox_inches='tight')

# Scenario C

fig, ax = plt.subplots(figsize=(18, 9))

ax.plot(ramp_caricoC['PwrTOT_rel'], label='Scenario C', linewidth=3, color='tab:red')
locs_y = ax.get_yticks()
ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
ax.set_ylabel('Potenza Relativa [%]')
ax.set_xlabel('Tempo [min]')
secax_y = ax.secondary_yaxis('right', functions=(rel2tot, tot2rel))
secax_y.set_ylabel(r'Potenza $[MW]$')

plt.savefig('IO/Opt_Pwr_base_scenarioC', bbox_inches='tight')
