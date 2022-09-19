import os
from cust_libs.data_processing import compute
from cust_libs.misc import gas_eur
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')

data_ON_path = 'data/processed/Corsini2021/Corsini2021_Processed_ON.csv'

data_ON = pd.read_csv(data_ON_path)

eta_avg = compute(data_ON, 'avg_eta')
eta_avg_2 = 0.54

pwr_tot = compute(data_ON, 'tot_en_out')

x = np.linspace(0.001, 0.009, 100, endpoint=True).reshape(-1, 1)

risparmio_gas = []
risparmio_gas_2 = []
for i in x:
    pwr_in_new = pwr_tot / (eta_avg + i)
    pwr_in_new_2 = pwr_tot / (eta_avg_2 + i)
    risparmio_gas.append(pwr_tot / eta_avg - pwr_in_new)
    risparmio_gas_2.append(pwr_tot / eta_avg_2 - pwr_in_new_2)

gas2eur, eur2gas = gas_eur(250)
fig, ax = plt.subplots(figsize=(18, 9))
ax.plot(x, risparmio_gas, label='vecchio')
ax.plot(x, risparmio_gas_2, label='nuovo')
ax.set_xlabel(r'Incremento rendimento')
ax.set_ylabel(r'Risparmio gas [MWh]')

secax_y = ax.secondary_yaxis('right', functions=(gas2eur, eur2gas))
secax_y.set_ylabel(r'Risparmio [M eur]')

ax.grid(color='black', which='major', axis='y', alpha=0.5, linestyle='dashed', linewidth=1)
ax.grid(color='black', which='minor', axis='y', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax.tick_params(axis='y', which='minor', left=True)
ax.axvline(x=0.005, color='r', linestyle='dashed', linewidth=1, alpha=0.5)

plt.legend(loc='best')
plt.show()