# LIBs
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.collections as collections
from scipy import stats
from numpy.polynomial import Polynomial

# Custom LIBs
from cust_libs.data_processing import filter_data, transf_fun

os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
data_accensione = 'data/processed/Corsini2021/acc/Corsini2021_Acc10.csv'

data = pd.read_csv(data_accensione)
mpl.rcParams["font.size"] = 18

rel2tot, tot2rel = transf_fun(data)

data['DateTime'] = pd.to_datetime(data['DateTime'])

x = data.index.values
pwr_rel = data['PwrTOT'].values
terna_load = data['Terna_Load']
avg_pwr = np.average(pwr_rel)

fig, ax = plt.subplots(figsize=(18, 9))

ax.plot(x, pwr_rel, linewidth=2, color='b')

ax2 = ax.twinx()
ax2.plot(x, terna_load, color='g', linewidth=2, alpha=0.5)

ax.set_xlim([0, len(data)])
ax.set_ylim([0, 0.9*410])
locs_y2 = ax2.get_yticks()
ax2.set_yticks(locs_y2, np.round(locs_y2 / 1000, 1))
ax.set_ylabel('Potenza [MW]')
ax.set_xlabel('Tempo [min]')
ax2.set_ylabel('Domanda di Rete [GW]')

plt.legend()
res = stats.pearsonr(pwr_rel, terna_load)
print(res)
# plt.savefig("temp/esempio_terna_02" + ".png", bbox_inches='tight')
