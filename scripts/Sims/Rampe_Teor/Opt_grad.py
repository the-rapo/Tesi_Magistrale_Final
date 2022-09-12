from cust_libs.sims import simple_ramp
from cust_libs.modeling import Load_ML_Model
from cust_libs.misc import transf_fun
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')

model, _ = Load_ML_Model('models/multivariate/ML/RND_FOR/RND_FOR_02.joblib')
# model, _ = Load_ML_Model('models/multivariate/ML/ALL/ALL_02.joblib')
p_nom = 410

# Ramp Carico
t_lowp = 2 * 60
t_highp = 2 * 60
t_ramp = 20

low_p = 0.47
high_p = 0.85

# Ramp BESS Ottimale (Simmetrico)
t_lowp_2 = 2 * 60 - 30
t_highp_2 = 2 * 60 - 30
t_ramp_2 = 80

low_p_2 = 0.47
high_p_2 = 0.85

# Ramp BESS Non Ottimale (Asimmetrico)
t_lowp_3 = 2 * 60
t_highp_3 = 2 * 60 - 60
t_ramp_3 = 80

low_p_3 = 0.47
high_p_3 = 0.85

# Ramp BOOST
t_lowp_4 = 2 * 60
t_highp_4 = 2 * 60 + 10
t_ramp_4 = 10

low_p_4 = 0.47
high_p_4 = 0.85


ramp_carico = simple_ramp(low_p, high_p, t_lowp, t_highp, t_ramp)
ramp_BESS_opt = simple_ramp(low_p_2, high_p_2, t_lowp_2, t_highp_2, t_ramp_2)
ramp_BESS_non_opt = simple_ramp(low_p_3, high_p_3, t_lowp_3, t_highp_3, t_ramp_3)
ramp_boost = simple_ramp(low_p_4, high_p_4, t_lowp_4, t_highp_4, t_ramp_4)


ramp_carico['Rendimento'] = model.predict(ramp_carico[['PwrTOT_rel', 'Grad_PwrTOT_rel']].values)
ramp_carico['Pwr_in'] = np.divide(ramp_carico['PwrTOT'], ramp_carico['Rendimento'])
ramp_carico['BESS_Pwr'] = np.subtract(ramp_carico['PwrTOT'], ramp_carico['PwrTOT'])
tot_energy_in_carico = sum(ramp_carico['Pwr_in'].values) / 60
tot_energy_out_carico = sum(ramp_carico['PwrTOT'].values) / 60
rendimento_carico = tot_energy_out_carico / tot_energy_in_carico
grad_carico = ramp_carico['Grad_PwrTOT_rel'].iloc[125] * p_nom
tot_energy_in_rampa_carico = sum(ramp_carico['Pwr_in'].iloc[120: 140].values) / 60
tot_energy_out_rampa_carico = sum(ramp_carico['PwrTOT'].iloc[120: 140].values) / 60
rendimento_rampa_carico = tot_energy_out_rampa_carico / tot_energy_in_rampa_carico


ramp_BESS_opt['Rendimento'] = model.predict(ramp_BESS_opt[['PwrTOT_rel', 'Grad_PwrTOT_rel']].values)
ramp_BESS_opt['Pwr_in'] = np.divide(ramp_BESS_opt['PwrTOT'], ramp_BESS_opt['Rendimento'])
ramp_BESS_opt['BESS_Pwr'] = np.subtract(ramp_carico['PwrTOT'], ramp_BESS_opt['PwrTOT'])
tot_energy_in_BESS_opt = sum(ramp_BESS_opt['Pwr_in'].values) / 60
tot_energy_out_BESS_opt = sum(ramp_BESS_opt['PwrTOT'].values) / 60
rendimento_BESS_opt = tot_energy_out_BESS_opt / tot_energy_in_BESS_opt
grad_BESS_opt = ramp_BESS_opt['Grad_PwrTOT_rel'].iloc[125] * p_nom
tot_energy_in_rampa_BESS_opt = sum(ramp_BESS_opt['Pwr_in'].iloc[90: 170].values) / 60
tot_energy_out_rampa_BESS_opt = sum(ramp_BESS_opt['PwrTOT'].iloc[90: 170].values) / 60
rendimento_rampa_BESS_opt = tot_energy_out_rampa_BESS_opt / tot_energy_in_rampa_BESS_opt


ramp_BESS_non_opt['Rendimento'] = model.predict(ramp_BESS_non_opt[['PwrTOT_rel', 'Grad_PwrTOT_rel']].values)
ramp_BESS_non_opt['Pwr_in'] = np.divide(ramp_BESS_non_opt['PwrTOT'], ramp_BESS_non_opt['Rendimento'])
ramp_BESS_non_opt['BESS_Pwr'] = np.subtract(ramp_carico['PwrTOT'], ramp_BESS_non_opt['PwrTOT'])
tot_energy_in_BESS_non_opt = sum(ramp_BESS_non_opt['Pwr_in'].values) / 60
tot_energy_out_BESS_non_opt = sum(ramp_BESS_non_opt['PwrTOT'].values) / 60
rendimento_BESS_non_opt = tot_energy_out_BESS_non_opt / tot_energy_in_BESS_non_opt
grad_BESS_non_opt = ramp_BESS_non_opt['Grad_PwrTOT_rel'].iloc[125] * p_nom
tot_energy_in_rampa_BESS_non_opt = sum(ramp_BESS_non_opt['Pwr_in'].iloc[120: 200].values) / 60
tot_energy_out_rampa_BESS_non_opt = sum(ramp_BESS_non_opt['PwrTOT'].iloc[120: 200].values) / 60
rendimento_rampa_BESS_non_opt = tot_energy_out_rampa_BESS_non_opt / tot_energy_in_rampa_BESS_non_opt


ramp_boost['Rendimento'] = model.predict(ramp_boost[['PwrTOT_rel', 'Grad_PwrTOT_rel']].values)
ramp_boost['Pwr_in'] = np.divide(ramp_boost['PwrTOT'], ramp_boost['Rendimento'])
tot_energy_in_boost = sum(ramp_boost['Pwr_in'].values) / 60
tot_energy_out_boost = sum(ramp_boost['PwrTOT'].values) / 60
rendimento_boost = tot_energy_out_boost / tot_energy_in_boost
grad_boost = ramp_boost['Grad_PwrTOT_rel'].iloc[125] * p_nom
tot_energy_in_rampa_boost = sum(ramp_boost['Pwr_in'].iloc[120: 130].values) / 60
tot_energy_out_rampa_boost = sum(ramp_boost['PwrTOT'].iloc[120: 130].values) / 60
rendimento_rampa_boost = tot_energy_out_rampa_boost / tot_energy_in_rampa_boost


# PLOTS
mpl.rcParams["font.size"] = 18
rel2tot, tot2rel = transf_fun(p_nom)

# FIGURA 1 - TIPI DI CURVE DI CARICO

fig, ax = plt.subplots(figsize=(18, 9))

ax.plot(ramp_BESS_opt['PwrTOT_rel'], label='BESS Ottimizzato', linewidth=2)
ax.plot(ramp_BESS_non_opt['PwrTOT_rel'], label='BESS non Ottimizzato', linewidth=2)
ax.plot(ramp_carico['PwrTOT_rel'], label='Carico', linewidth=2, linestyle='dashed')
ax.axvline(x=120, color='k', linestyle=(0, (5, 10)), linewidth=2, label='Segnale Variazione Carico')
locs_y = ax.get_yticks()
ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
ax.set_ylabel('Potenza Relativa [%]')
ax.set_xlabel('Tempo [min]')
secax_y = ax.secondary_yaxis('right', functions=(rel2tot, tot2rel))
secax_y.set_ylabel(r'Potenza $[MW]$')
plt.legend(loc='best')

plt.savefig('IO/Opt_Grad_ConfrontoCarichi', bbox_inches='tight')
# FIGURA 2

fig, ax = plt.subplots(figsize=(18, 9))

ax.plot(ramp_BESS_opt['BESS_Pwr'], label='BESS Ottimizzato', linewidth=2)
ax.plot(ramp_BESS_non_opt['BESS_Pwr'], label='BESS non Ottimizzato', linewidth=2)
ax.axvline(x=120, color='k', linestyle=(0, (5, 10)), linewidth=2, label='Segnale Variazione Carico')
plt.legend(loc='best')
ax.set_ylabel(r'Potenza BESS $[MW]$')
ax.set_xlabel('Tempo [min]')
plt.savefig('IO/Opt_Grad_BESS_PWR', bbox_inches='tight')

# FIGURA 3 - boost

fig, ax = plt.subplots(figsize=(18, 9))

ax.plot(ramp_carico['PwrTOT_rel'], label=r'8 $MW/min$', linewidth=2)
ax.plot(ramp_boost['PwrTOT_rel'], label=r'16 $MW/min$', linewidth=2)
ax.axvline(x=120, color='k', linestyle=(0, (5, 10)), linewidth=2, label='Segnale Variazione Carico')
locs_y = ax.get_yticks()
ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
ax.set_ylabel('Potenza Relativa [%]')
ax.set_xlabel('Tempo [min]')
secax_y = ax.secondary_yaxis('right', functions=(rel2tot, tot2rel))
secax_y.set_ylabel(r'Potenza $[MW]$')
plt.legend(loc='best')

plt.savefig('IO/Opt_Grad_Boost', bbox_inches='tight')


grad_i = []
rend_i = []
rend_i_rampa = []
for i in range(0, 33):
    t_ramp_i = 40 - i
    t_lowp_i = 2 * 60
    t_highp_i = 260 - 120 - t_ramp_i

    low_p_i = 0.47
    high_p_i = 0.85

    ramp_iter = simple_ramp(low_p_i, high_p_i, t_lowp_i, t_highp_i, t_ramp_i)
    ramp_iter['Rendimento'] = model.predict(ramp_iter[['PwrTOT_rel', 'Grad_PwrTOT_rel']].values)
    ramp_iter['Pwr_in'] = np.divide(ramp_iter['PwrTOT'], ramp_iter['Rendimento'])
    tot_energy_in_iter = sum(ramp_iter['Pwr_in'].values) / 60
    tot_energy_in_iter_rampa = sum(ramp_iter['Pwr_in'].iloc[120: 120 + t_ramp_i].values) / 60
    tot_energy_out_iter = sum(ramp_iter['PwrTOT'].values) / 60
    tot_energy_out_iter_rampa = sum(ramp_iter['PwrTOT'].iloc[120: 120 + t_ramp_i].values) / 60
    rendimento_iter = tot_energy_out_iter / tot_energy_in_iter
    rendimento_iter_rampa = tot_energy_out_iter_rampa / tot_energy_in_iter_rampa

    grad_iter = ramp_iter['Grad_PwrTOT_rel'].iloc[125] * p_nom
    grad_i.append(grad_iter)
    rend_i.append(rendimento_iter)
    rend_i_rampa.append(rendimento_iter_rampa)

# FIGURA 4 - gradiente

fig, ax = plt.subplots(figsize=(18, 9))

ax.plot(grad_i, rend_i, linewidth=2, label='Rendimento complesivo')
ax.plot(grad_i, rend_i_rampa, linewidth=2, label='Rendimento del transitorio')
locs_y = ax.get_yticks()
ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
ax.set_ylabel('Rendimento [%]')
ax.set_xlabel(r'$\nabla P$ [MW/min]')
plt.legend(loc='best')
plt.savefig('IO/Opt_Grad_Boost_Confronto', bbox_inches='tight')

# FIGURA 5 - TIPI DI CURVE DI CARICO

fig, ax = plt.subplots(figsize=(18, 9))

ax.plot(ramp_BESS_opt['Rendimento'], label='BESS Ottimizzato', linewidth=2)
ax.plot(ramp_BESS_non_opt['Rendimento'], label='BESS non Ottimizzato', linewidth=2)
ax.plot(ramp_carico['Rendimento'], label='Carico', linewidth=2)
ax.axvline(x=120, color='k', linestyle=(0, (5, 10)), linewidth=2, label='Segnale Variazione Carico')
locs_y = ax.get_yticks()
ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
ax.set_ylabel('Rendimento [%]')
ax.set_xlabel('Tempo [min]')

plt.legend(loc='best')

plt.savefig('IO/Opt_Grad_ConfrontoCarichi_rendimento', bbox_inches='tight')
'''
ramp.plot()
ramp2.plot()
plt.show()

ramp_carico['Rendimento'].plot()
ramp2['Rendimento'].plot()


plt.show()
'''