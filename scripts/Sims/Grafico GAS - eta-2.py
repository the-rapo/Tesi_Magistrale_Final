import os
from cust_libs.data_processing import compute
from cust_libs.misc import gas_eur
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')

q_elec = 1800 * 1000 # MWh

eta = 0.547 * 0.9233

d_eta = 0.005

delta_gas = - q_elec * (1 / (eta + d_eta) - 1 / eta)
delta_gas_graf = []
x = np.linspace(0.49, 0.55, 100, endpoint=True).reshape(-1, 1)
for i in x:
    delta_gas_graf.append(- q_elec * (1 / (i + d_eta) - 1 / i))

mpl.rcParams["font.size"] = 18
gas2eur, eur2gas = gas_eur(200)
fig, ax = plt.subplots(figsize=(18, 9))
ax.plot(x, delta_gas_graf, label='nuovo')
ax.set_xlabel(r'Rendimento Base [%]')
ax.set_ylabel(r'Risparmio Gas [MWh]')

secax_y = ax.secondary_yaxis('right', functions=(gas2eur, eur2gas))
secax_y.set_ylabel(r'Risparmio [M eur]')

ax.grid(color='black', which='major', axis='y', alpha=0.5, linestyle='dashed', linewidth=1)
ax.grid(color='black', which='minor', axis='y', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax.tick_params(axis='y', which='minor', left=True)
ax.axvline(x=eta, color='r', linestyle='dashed', linewidth=1, alpha=0.5)
locs_x = ax.get_xticks()
ax.set_xticks(locs_x, np.round(locs_x * 100, 1))

plt.show()

